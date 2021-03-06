from re import S
from turtle import home
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split


sample_size = 5000
min_rating_v = 3
min_rating_cnt = 5
max_rating_cnt = 30

# read ratings
ratings = pd.read_csv('~/data/movielens/ml-25m/ratings.csv')
print(ratings)
print(ratings.describe())

# filter out by min rating value
rating_capped =  ratings.loc[ratings['rating'] > min_rating_v]
print(rating_capped.describe())

# count ratings per user
users = rating_capped.groupby(['userId']).size().reset_index(name="cnt")
print(users.describe())

# filter out by ratings count
users_capped =  users.loc[(users['cnt'] >= min_rating_cnt) & (users['cnt'] <= max_rating_cnt)]
print(users_capped.describe())

# draw sample
users_capped_sample = users_capped.sample(sample_size)
print(users_capped_sample.describe())

# select ratings for sampled users
ratings_sample = pd.merge(rating_capped, users_capped_sample, on=['userId'],how='inner')
ratings_sample = ratings_sample.loc[(ratings_sample['cnt'] >= min_rating_cnt) & (ratings_sample['cnt'] <= max_rating_cnt)]
ratings_sample.drop(columns=['cnt'], inplace=True)
print(ratings_sample)
print(ratings_sample.describe())


# sort
ratings_sample = ratings_sample.sort_values(by=['userId', 'timestamp'])
print(ratings_sample.head(100))

# shift ts
ratings_sample["timestamp"] = ratings_sample["timestamp"].sub(ratings_sample["timestamp"].max()) 
ratings_sample["timestamp"] = ratings_sample["timestamp"].apply(lambda x : int(-x/60))

# check
print(ratings_sample.groupby(['userId']).size().reset_index(name="cnt").describe())

# covert items col to string of items
ratings_sample=ratings_sample.groupby('userId').agg(lambda x: list(x))
ratings_sample["movies"]=ratings_sample['movieId'].apply(lambda x: '|'.join(map(str,x)))
ratings_sample["timestamps"]=ratings_sample['timestamp'].apply(lambda x: '|'.join(map(str,x)))
ratings_sample.drop(columns=['movieId', 'timestamp'],inplace=True)

# split into train and test
rs = check_random_state(1977)
train, test = train_test_split(ratings_sample, test_size = 0.3, random_state=rs)

# write down
train.to_csv(f'~/data/movielens/ml-25m/train_{sample_size}_{min_rating_v}_{min_rating_cnt}_{max_rating_cnt}.csv', index=False)
test.to_csv(f'~/data/movielens/ml-25m/test_{sample_size}_{min_rating_v}_{min_rating_cnt}_{max_rating_cnt}.csv', index=False)