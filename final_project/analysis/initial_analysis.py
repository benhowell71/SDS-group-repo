import pandas as pd
import numpy as np
from pyparsing import col
from requests import head
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from torch import rand
import seaborn as sns
import matplotlib.pyplot as plt

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
# okay there are a few values that don't have a Western tag I'm filling it w/ a 0
# looks like they're all the same movie, item_id which is unknown so I'm actually just gonna remove that from this set
data[data.Western.isna()]

data = data[data.Western.notna()].astype({'Western': 'int'})
# now everything can be nice, neat integers
data.gender.unique()
data['gender_id'] = np.where(data.gender == 'M', 0, 1)

features = ['rating', 'age', 'gender_id', 'job', 'unknown', 'Action', 'Adventure', 'Animation','Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western']

data_x = data[features].drop(columns=['rating'])
data_y = data[['rating']]

train_x, train_y, test_x, test_y = train_test_split(data_x, data_y, test_size=0.33, random_state=123)

model = LinearRegression().fit(train_x, test_x)

r_sq = round(model.score(train_x, test_x), 3)
print('coefficient of determination training set:', r_sq)

r_sq2 = round(model.score(train_y, test_y), 3)
print('coefficient of determination training set:', r_sq2)

model.coef_

pred = pd.DataFrame(model.predict(train_y), columns=['pred_rating'])
test_data = train_y.join(test_y).reset_index().join(pred)

sns.scatterplot(data=test_data, x = 'rating', y = 'pred_rating', hue = 'pred_rating')
sns.set(style='whitegrid',)
plt.show()

sns.boxplot(data=test_data, x = 'rating', y = 'pred_rating')
sns.set(style='whitegrid',)
plt.show()

sns.boxplot(data=test_data, x = 'gender_id', y = 'pred_rating')
sns.set(style='whitegrid',)
plt.show()