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
n_epochs   = 200
lr         = 0.1
dropout    = 0.1 

# debug flag
debug = False


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
    def __init__(self, embed_dim, num_layers, num_items, batch_size, dropout, max_sequence_length, pad_index =0, debug = False):
        super(RnnRecommender,self).__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.hidden_size = embed_dim 
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.pad_index = pad_index
        self.debug = debug

        # Embedding layer
        self.embeds = nn.Embedding(num_items, self.embed_dim, padding_idx=pad_index)

        # LSTM layers
        self.lstm = nn.LSTM(batch_first = True, input_size=self.embed_dim, hidden_size=self.hidden_size, num_layers = self.num_layers, dropout=dropout)

        # Linear layer
        self.item2item = nn.Linear(self.hidden_size, num_items)


    def init_hidden(self):        
        '''
        Reset of LSTM cells
        '''

        hidden_state = torch.randn(self.num_layers, self.batch_size, self.hidden_size)
        cell_state = torch.randn(self.num_layers, self.batch_size, self.hidden_size)

        return (Variable(hidden_state), Variable(cell_state))


    def forward(self, inputs, inputs_true_size):        
         
        self.hidden = self.init_hidden()
        embeds = self.embeds(inputs)
        embeds_packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, inputs_true_size, batch_first=True, enforce_sorted=False)        
        lstm_out_packed, _ = self.lstm(embeds_packed)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out_packed, batch_first=True, padding_value= self.pad_index, total_length=self.max_sequence_length)
        item_space = self.item2item(lstm_out)
        item_scores = F.log_softmax(item_space, dim=1)

        # reshape to match with labeles shape
        item_scores = item_scores.view(-1,item_scores.shape[2])

        if self.debug:
            print(f"embeds {embeds.shape}")
            #print(f"embeds_packed {embeds_packed.shape}")
            #print(f"lstm_out_packed {lstm_out_packed.shape}")            
            print(f"lstm_out {lstm_out.shape}")
            print(f"item_space {item_space.shape}")
            print(f"item_scores {item_scores.shape}")
            print(f"item_scores.view {item_scores.shape}")

        return item_scores

# Train steps within one epoch
def train_epoch(loss_function, optimizer, model, data):

    # Keep track of the total loss for the batch
    total_loss = 0
    for inputs, labels, inputs_true_size in data:
        # Clear the gradients
        optimizer.zero_grad()
        # Run a forward pass    
        outputs = model.forward(inputs, inputs_true_size)        
        labels_v = labels.view(-1)
        # Compute the batch loss
        if model.debug:
            print(f"outputs {outputs.shape}")
            print(f"labels {labels.shape}")
            print(f"labels_v {labels_v.shape}")

        loss = loss_function(outputs, labels_v)
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
    print(f"train data size: {len(ratings)}")

    m, t, l = split_input_data(ratings)

    # get number of items per sequence    
    m_size = [len(items) for items in m]
    max_sequence_length = max(m_size)
    print(f"max n of items in sequence: {max_sequence_length}")
    m_size = torch.tensor(m_size)
    

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

    data = list(zip(m_ixs_padded,l_ixs_padded, m_size))
    dataloader = DataLoader(data, batch_size = batch_size, shuffle=True) 

    # Train model
    model = RnnRecommender(embed_dim, num_layers, len(unique_m), batch_size, dropout, max_sequence_length, pad_index=item_to_ix[pad_token], debug = debug)
    loss_function = nn.CrossEntropyLoss(ignore_index=item_to_ix[pad_token])
    optimizer = optim.SGD(model.parameters(), lr=lr)
    tr_start = datetime.now()
    train(loss_function, optimizer, model, dataloader, n_epochs)
    print(f"training time: {datetime.now() - tr_start}")

    # Save trained model for later predictons
    torch.save(model.state_dict(), f'rrn_rec_lstm_batch_{batch_size}_epochs_{n_epochs}_{start.strftime("%m%d%Y_%H%M%S")}.pt')