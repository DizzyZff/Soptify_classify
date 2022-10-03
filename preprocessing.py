import pandas as pd
import numpy as np
import sqlite3

data_path = 'musicData.csv'
data = pd.read_csv(data_path)
df = pd.DataFrame(data)
df = df.dropna()
df = df.replace("?", -1)
#str to int
artist_name = df['artist_name'].dropna()
df['artist_name'] = artist_name.replace(artist_name.unique(), np.arange(0, len(artist_name.unique())))
song_name = df['track_name'].dropna()
df['track_name'] = song_name.replace(song_name.unique(), np.arange(0, len(song_name.unique())))
key = df['key'].dropna().unique()
df['key'] = df['key'].replace(key, np.arange(0, len(key)))
#dummy code mode
mode = df['mode'].dropna().unique()
df['mode'] = df['mode'].replace(mode, np.arange(0, len(mode)))
obtained_date = df['obtained_date'].dropna().unique()
df['obtained_date'] = df['obtained_date'].replace(obtained_date, np.arange(0, len(obtained_date)))
music_genre = df['music_genre'].dropna().unique()
df['music_genre'] = df['music_genre'].replace(music_genre, np.arange(0, len(music_genre)))

#to sql
conn = sqlite3.connect('musicData.db')
c = conn.cursor()
df.to_sql('musicData', conn, if_exists='replace', index = False)
conn.commit()
conn.close()

print('finish')

