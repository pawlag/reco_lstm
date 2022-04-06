# reco_lstm - recommender model using deep learning LSTM network 

Recommender system based on LSTM.\
Implemented in pytorch.\
Applied layers: embedding, LSTM (stacked), linear\
Regularization: dropout in LSTM cells \
Trained on 50k (in terms of unique users) data sample from movie lens 25M dataset https://grouplens.org/datasets/movielens/25m/. 
The following sampling was applied in order to reduce data size and includ in ratings series only positive ones: i) only movies with ratings above 3 (positive), ii) only ratings for users having number of positive rating ranging from 3 to 30   




## contents:
- rnn_recommender_lstm_batched.py - model definiton and training loop
- data_prep.py - movie lens preprocessing (sampling)  
