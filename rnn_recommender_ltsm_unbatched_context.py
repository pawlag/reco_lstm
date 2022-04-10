from random import shuffle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from functools import partial
from datetime import datetime
import pickle

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


# Configurations 

# train data
train_data_path = '/home/luser/data/movielens/ml-25m/train_5000_3_5_30.csv'

# model hyperparameters
embed_dim= 64
lstm_hidden_size = 128
lstm_num_layers=3
n_epochs = 100
lr = 0.1

# number of time bins (+1 left for padding)
num_time_bins = 99

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
    def __init__(self, embed_dim, lstm_num_layers, lstm_hidden_size, num_items, num_time_bins, pad_value =0, debug = False):
        super(RnnRecommender,self).__init__()

        self.embed_dim = embed_dim        
        self.num_layers = lstm_num_layers
        self.hidden_size = lstm_hidden_size

        # Embedding layer
        self.embeds_items = nn.Embedding(num_items, self.embed_dim, padding_idx=pad_value)
        self.embeds_time = nn.Embedding(num_time_bins, self.embed_dim, padding_idx=0)

        # LSTM layers
        self.lstm = nn.LSTM(batch_first = True, input_size=2*self.embed_dim, hidden_size=self.hidden_size, num_layers = self.num_layers, dropout=0.1)

        # Linear layer
        self.item2item = nn.Linear(2*self.embed_dim, num_items)

        # debug flag
        self.debug = debug
    
    def forward(self, items, time):
        embeds_items = self.embeds_items(items)
        embeds_time = self.embeds_time(time)
        embeds = torch.cat([embeds_items,embeds_time],1)        
        if self.debug:
            print(f"embeds_items {embeds_items.shape}")            
            print(f"embeds_time {embeds_time.shape}")
            print(f"embeds {embeds.shape}")            

        lstm_out, _ = self.lstm(embeds)
        if self.debug:
            print(f"lstm_out {lstm_out.shape}")

        item_space = self.item2item(lstm_out.view(len(items), -1))
        if self.debug:
            print(f"item_space {item_space.shape}")

        item_scores = F.log_softmax(item_space, dim=1)
        if self.debug:
            print(f"item_scores {item_scores.shape}")
        
        return item_scores

# Train steps within one epoch
def train_epoch(loss_function, optimizer, model, data):

    # Keep track of the total loss for the batch
    total_loss = 0
    for items, labels, time in data:
        # Clear the gradients
        optimizer.zero_grad()
        # Run a forward pass    
        outputs = model.forward(items, time)
        # Compute the batch loss
        loss = loss_function(outputs, labels)
        # Calculate the gradients
        loss.backward()
        # Update the parameteres
        optimizer.step()
        total_loss += loss.item()

    return total_loss


# Function containing  main training loop through all epoches
def train(loss_function, optimizer, model, data, num_epochs=100):

    # Iterate through each epoch and call our train_epoch function
    for i, epoch in enumerate(range(num_epochs)):
        epoch_loss = train_epoch(loss_function, optimizer, model, data)
        if epoch % 1 == 0: 
            print(f"epoch {i}: loss: {epoch_loss}")

def discretize_time(t, n_bins):

    # get t values as long list
    t_array = np.array([x for y in t for x in y]).reshape(-1,1)
    discretizer = KBinsDiscretizer(n_bins,encode='ordinal').fit(t_array)

    for i, _ in enumerate(t):        
        t[i]= discretizer.transform(np.array(t[i]).reshape(-1,1)).reshape(-1)
        t[i] = t[i]+1
    return t , discretizer

# Run
if __name__ == "__main__":

    start = datetime.now()

    # Data
    ratings = pd.read_csv(train_data_path)
    ratings = ratings.to_dict(orient='records')

    m, t, l = split_input_data(ratings)

    _max_ratings_cnt = 0
    for items in m:
        if len(items)>_max_ratings_cnt:
            _max_ratings_cnt = len(items)
    print(f"max ratings count {_max_ratings_cnt}")

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

    # Convert time for bins
    t, dicretizer = discretize_time(t, num_time_bins)

    with open(f'dicretizer_{start.strftime("%m%d%Y_%H%M%S")}.pcl','wb') as f:
        pickle.dump(dicretizer,f)

    # Apply padding to have equal size input sequences  
    m_ixs = [torch.LongTensor(i) for i in m_ixs]
    m_ixs_padded = nn.utils.rnn.pad_sequence(m_ixs, batch_first=True, padding_value=item_to_ix[pad_token])

    l_ixs = [torch.LongTensor(i) for i in l_ixs]
    l_ixs_padded = nn.utils.rnn.pad_sequence(l_ixs, batch_first=True, padding_value=item_to_ix[pad_token])

    t = [torch.LongTensor(i) for i in t]
    t_padded = nn.utils.rnn.pad_sequence(t, batch_first=True, padding_value=0)

    data = list(zip(m_ixs_padded,l_ixs_padded, t_padded))

    # Train model
    model = RnnRecommender(embed_dim, lstm_num_layers, lstm_hidden_size, len(unique_m), num_time_bins+1, debug=debug)
    loss_function = nn.CrossEntropyLoss(ignore_index=item_to_ix[pad_token])
    optimizer = optim.SGD(model.parameters(), lr=lr)
    tr_start = datetime.now()
    train(loss_function, optimizer, model, data, n_epochs)
    print(f"training time: {datetime.now() - tr_start}")

    # Save trained model for later predictons
    torch.save(model.state_dict(),f'rrn_rec_lstm_nbatch_{n_epochs}_{start.strftime("%m%d%Y_%H%M%S")}.pt')