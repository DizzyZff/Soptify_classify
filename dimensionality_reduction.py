import pandas as pd
import sqlite3
from sklearn.decomposition import PCA
import plotly.express as px

#load from db
data_path = 'musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
df = pd.read_sql_query("SELECT * FROM musicData", conn)
conn.close()
print("Finish Loading")

#extract cartagorical data
dropped_df = df.drop(['music_genre'], axis=1)
dropped_df = dropped_df.drop(['mode'], axis=1)

#PCA
pca = PCA(n_components=3)
pca.fit(dropped_df)
pca_result = pca.transform(dropped_df)
pca_result = pd.DataFrame(pca_result, columns=['pca1', 'pca2', 'pca3'])
pca_result['mode'] = df['mode']
pca_result['music_genre'] = df['music_genre']


#plot
fig = px.scatter_3d(pca_result, x='pca1', y='pca2', z='pca3', color='music_genre')
fig.show()

#to sql
conn = sqlite3.connect('musicData.db')
c = conn.cursor()
pca_result.to_sql('pca_result', conn, if_exists='replace', index = False)
conn.commit()
conn.close()

print('Finish updating pca_result datasets.')

