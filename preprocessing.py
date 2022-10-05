import pandas as pd
import numpy as np
import sqlite3

#
data_path = 'musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
df = pd.read_sql_query("SELECT * FROM musicData", conn)
conn.close()
print("Finish Loading")

print(df.head())
print(df.info())
print(df.describe())

# check missing value
missing_count = df.isnull().sum()
value_count = df.isnull().count()
missing_rate = round(missing_count / value_count * 100, 2)
missing_df = pd.DataFrame({'missing_count': missing_count, 'missing_rate': missing_rate})
print(missing_df)
df = df.dropna()
# replace ? with -1
df = df.replace('?', -1)

# name and title to numeric
df['artist_name'] = df['artist_name'].apply(len)
df['track_name'] = df['track_name'].apply(len)

# obtained date remove month
obtain_date = df['obtained_date'].unique()
df['obtained_date'] = df['obtained_date'].replace(obtain_date, range(len(obtain_date)))

# mode to dummy code
df = pd.get_dummies(df, columns=['mode'])

# key to dummy code
key = df['key'].unique()
df['key'] = df['key'].replace(key, range(len(key)))

# music genre to numeric
factor = pd.factorize(df['music_genre'])
df['music_genre'] = factor[0]
definitions = factor[1]
print(df['music_genre'].head())
print(definitions)

# TEMPO to numeric
df['tempo'] = df['tempo'].astype(float)

df = df.drop(['artist_name'], axis=1)
df = df.drop(['track_name'], axis=1)
"""df = df.drop(['instance_id'], axis=1)
df = df.drop(['obtained_date'], axis=1)
""""""
df = df.drop(['artist_name_length'], axis=1)
df = df.drop(['track_name_length'], axis=1)"""


print(df.head())
print(df.info())
print(df.describe())
print(df['tempo'].describe())

# to sql
conn = sqlite3.connect('musicData.db')
c = conn.cursor()
df.to_sql('musicData_clean', conn, if_exists='replace', index=False)
conn.commit()
conn.close()

# to csv
df.to_csv('musicData_clean.csv', index=False)
