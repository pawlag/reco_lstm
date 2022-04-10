# reco_lstm - recommender model using deep learning LSTM network 

Recommender system based on LSTM.\
Implemented in PyTorch.\
Applied layers: embedding, LSTM (stacked), linear\
Regularization: dropout in LSTM cells \
Trained on 50k (in terms of unique users) data sample from movie lens 25M dataset https://grouplens.org/datasets/movielens/25m/. 
The following sampling was applied in order to reduce data size: i) only movies with ratings above 3 (positive), ii) only ratings for users having number of positive ratings ranging from 3 to 30. Reasoning for such approach is to get balanced sample expressing user actions related to positive decisions.       


## contents:
- rnn_recommender_lstm_batched.py - model definiton and training based on batched sequnce data, include packing and unpacking of padded sequance juest before and after LSTM layer
- rnn_recommender_lstm_unbatched.py - model definiton and training sequance by sequence    
- rnn_recommender_ltsm_unbatched_context.py - model with with rating time used as context feature (discretized / binned)
- data_prep.py - movie lens preprocessing (filtering, sampling and split to train and test sets)  