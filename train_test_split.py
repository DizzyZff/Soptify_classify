import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split

#load from db
data_path = 'musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
df = pd.read_sql_query("SELECT * FROM pca_result", conn)
conn.close()
print("Finish Loading")

# to 10 different genres datasets
genre_list = df['music_genre'].unique()
genre_list = genre_list.tolist()

datasets = []
for genre in genre_list:
    datasets.append(df[df['music_genre'] == genre])

train_datasets = []
test_datasets = []
for dataset in datasets:
    train, test = train_test_split(dataset, test_size=0.1, random_state=18402254)
    train_datasets.append(train)
    test_datasets.append(test)

# merge train datasets
train = pd.concat(train_datasets)
test = pd.concat(test_datasets)

print(train.shape, test.shape)

#to sql
conn = sqlite3.connect('musicData.db')
c = conn.cursor()
train.to_sql('train', conn, if_exists='replace', index = False)
test.to_sql('test', conn, if_exists='replace', index = False)
conn.commit()
conn.close()

print('Finish updating train and test datasets.')