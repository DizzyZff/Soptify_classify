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

#extract categorical data
categorical = df.drop(['music_genre'], axis=1)
categorical = categorical.drop(['mode'], axis=1)

#dimensionality reduction
pca = PCA(n_components=3)
pca.fit(categorical)
transformed = pca.transform(categorical)
plt = px.scatter_3d(transformed, x=0, y=1, z=2, color=df['music_genre'])
plt.show()

plt = px.scatter(transformed, x=0, y=1, color=df['music_genre'])
plt.show()
print('finish')

