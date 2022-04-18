import pandas as pd
import numpy as np
from requests import head
from sklearn.linear_model import LinearRegression

df = pd.read_csv('final_project\\data\\ml-100k\\u.data', sep='\t', header=None)

df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
# user id, age, gender, occupation, zip code
users = pd.read_csv('final_project\\data\\ml-100k\\u.user', sep='|', header=None)
users.columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']

genre = pd.read_csv('final_project\\data\\ml-100k\\u.genre', sep='|', header=None)
genre.columns = ['genre', 'genre_id']

job = pd.read_csv('final_project\\data\\ml-100k\\u.occupation', sep='|', header=None)
job.columns = ['occupation']
job = job.reset_index().rename(columns={'index': 'job'})

# before you run this, if you've newly downloaded the data
# do a ctrl+f for '||' and replace with '|' bc otherwise it doesn't work
items = pd.read_csv('final_project\\data\\ml-100k\\u.item', sep='|', header=None)
# items[~items.iloc(21) == 'unknown'] ignore
items.columns = ['movie_id', 'movie_title', 'video_release_date', 'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation','Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western']

data = df.merge(users, how='left', on = 'user_id').merge(job, how = 'left', on='occupation').merge(items, how='left', left_on='item_id', right_on='movie_id')