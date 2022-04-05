# reco_lstm - recommender model using deep learning LSTM netowork 

Recommender system based on LSTM.\
Implemented in pytorch.\
Applied layers: embedding, LSTM (stacked), linear\
Regularization: dropout in LSTM cells \
Trained on sample from movie lens 25M data https://grouplens.org/datasets/movielens/25m/. The following sampling was applied in order to reduce data size and includ in ratings series only positive ones     



## conent:
- rnn_recommender_lstm_batched.py - model definiton and training loop
- data_prep.py - movie lens preprocessing (sampling)  



