import pandas as pd
import sqlite3
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#load from db
data_path = 'musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
df = pd.read_sql_query("SELECT * FROM musicData", conn)
conn.close()
print("Finish Loading")

"""pca = PCA(n_components=3)
pca.fit(df)
transformed = pca.transform(df)
plt = px.scatter_3d(transformed, x=0, y=1, z=2, color=df['music_genre'])
plt.show()
"""
#t-SNE

tsne = TSNE(n_components=3)
transformed = tsne.fit_transform(df)
plt = px.scatter_3d(transformed, x=0, y=1, z=2, color=df['music_genre'])
plt.show()

plt = px.scatter(transformed, x=0, y=1, color=df['music_genre'])
plt.show()
print('finish')
