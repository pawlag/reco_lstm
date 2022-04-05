from random import shuffle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from functools import partial
from datetime import datetime
import pickle


# Configuration

# train data
train_data_path = '/home/luser/data/movielens/ml-25m/ratings_sample.csv'
batch_size = 128

# model hyperparameters
embed_dim  = 128
num_layers = 3
n_epochs   = 100
lr         = 0.1
dropout    = 0.1 


# Define utility functions 
def split_input_data(ratings):
    m=[]
    l=[]
    t=[]
    for item in ratings:
        movies = [int(x) for x in item['movies'].split('|')]
        ts    = [int(x) for x in item['timestamps'].split('|')]
        m.append(movies[:-1])
        l.append(movies[1:])
        t.append(ts[:-1])
    return m, t, l

# Define model
class RnnRecommender(nn.Module):
    def __init__(self, embed_dim, num_layers, num_items, batch_size, dropout, pad_index =0):
        super(RnnRecommender,self).__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.hidden_size = embed_dim 
        self.batch_size = batch_size

        # Embedding layer
        self.embeds = nn.Embedding(num_items, self.embed_dim, padding_idx=pad_index)

        # LSTM layers
        self.lstm = nn.LSTM(batch_first = True, input_size=self.embed_dim, hidden_size=self.hidden_size, num_layers = self.num_layers, dropout=dropout)

        # Linear layer
        self.item2item = nn.Linear(self.hidden_size, num_items)


    def init_hidden(self):        

        hidden_state = torch.randn(self.num_layers, self.batch_size, self.hidden_size)
        cell_state = torch.randn(self.num_layers, self.batch_size, self.hidden_size)

        return (Variable(hidden_state), Variable(cell_state))


    def forward(self, inputs):
        
        self.hidden = self.init_hidden()

        embeds = self.embeds(inputs)
        lstm_out, _ = self.lstm(embeds)
        item_space = self.item2item(lstm_out)
        item_scores = F.log_softmax(item_space, dim=1)
        # reshape to match with labeles shape
        item_scores = item_scores.view(-1,item_scores.shape[2])
        return item_scores

# Train steps within one epoch
def train_epoch(loss_function, optimizer, model, data):

    # Keep track of the total loss for the batch
    total_loss = 0
    for inputs, labels in data:
        # Clear the gradients
        optimizer.zero_grad()
        # Run a forward pass    
        outputs = model.forward(inputs)
        # Compute the batch loss
        loss = loss_function(outputs, labels.view(-1))
        # Calculate the gradients
        loss.backward()
        # Update the parameteres
        optimizer.step()
        total_loss += loss.item()

    return total_loss

# Main training loop through all epoches
def train(loss_function, optimizer, model, data, num_epochs=100):

    # Iterate through each epoch and call our train_epoch function
    for i, epoch in enumerate(range(num_epochs)):
        epoch_loss = train_epoch(loss_function, optimizer, model, data)
        if epoch % 1 == 0: 
            print(f"epoch {i} loss: {epoch_loss}")

# Run
if __name__ == "__main__":

    start = datetime.now()

    # Get training data
    ratings = pd.read_csv(train_data_path)
    ratings = ratings.to_dict(orient='records')

    m, t, l = split_input_data(ratings)

    _max_ratings_cnt = 0
    for items in m:
        if len(items)>_max_ratings_cnt:
            _max_ratings_cnt = len(items)
    print(f"max items in sequence count: {_max_ratings_cnt}")

    # Build set of unique items 
    def get_unique_items(m):
        return set(m for mvs in m for m in mvs)

    unique_m = get_unique_items(m)
    print(f"number of unique items: {len(unique_m)}")

    # Add pad token to items set
    pad_token = -2
    unique_m.add(pad_token)    

    # Add unknown token to items set
    unknown_token = -1
    unique_m.add(unknown_token)

    # Get item to index map
    ix_to_item = sorted(list(unique_m))
    item_to_ix = {item: ix for ix, item in enumerate(ix_to_item)}

    # Save ix_to_item map, handful for later predictions offline 
    with open(f'item_to_ix_{start.strftime("%m%d%Y_%H%M%S")}.pcl','wb') as f:
        pickle.dump(item_to_ix,f)


    # Convert items to indexes
    def convert_items_to_indices(items, item_to_ix, unknown_token):
        return [item_to_ix.get(item,item_to_ix[unknown_token]) for item in items]

    converter = partial(convert_items_to_indices, item_to_ix=item_to_ix, unknown_token=unknown_token)    
    m_ixs = [converter(items) for items in m]
    l_ixs = [converter(items) for items in l]


    # Apply padding to have equal size input sequences  
    m_ixs = [torch.LongTensor(x_i) for x_i in m_ixs]
    m_ixs_padded = nn.utils.rnn.pad_sequence(m_ixs, batch_first=True, padding_value=item_to_ix[pad_token])

    l_ixs = [torch.LongTensor(y_i) for y_i in l_ixs]
    l_ixs_padded = nn.utils.rnn.pad_sequence(l_ixs, batch_first=True, padding_value=item_to_ix[pad_token])

    data = list(zip(m_ixs_padded,l_ixs_padded))

    dataloader = DataLoader(data, batch_size = batch_size, shuffle=True) 


    # Train model
    model = RnnRecommender(embed_dim, num_layers, len(unique_m), batch_size, dropout, pad_index=item_to_ix[pad_token])
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    tr_start = datetime.now()
    train(loss_function, optimizer, model, dataloader, n_epochs)
    print(f"training time: {datetime.now() - tr_start}")

    # Save trained model for later predictons
    torch.save(model.state_dict(), f'rrn_rec_lstm_nbatch_{n_epochs}_{start.strftime("%m%d%Y_%H%M%S")}.pt')