import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import matplotlib.pyplot as plt1

# from db
from sklearn import datasets
from sklearn.decomposition import PCA

data_path = 'musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
df = pd.read_sql_query("SELECT * FROM musicData_clean", conn)
conn.close()
print("Finish Loading")
X = df.drop(['music_genre'], axis=1)
y = df['music_genre']

# correlation matrix
corr = df.corr()
print(corr)

# save to html
plt.write_html("corr.html")

# find eigenvalues and eigenvectors
pca = PCA(n_components=3, whiten=True)
pca.fit(X)
pca_result = pca.transform(X)
pca_result = pd.DataFrame(pca_result, columns=['pca1', 'pca2', 'pca3'])
pca_result['music_genre'] = y
pca_result['music_genre'] = pca_result['music_genre'].astype('category')

# plot
fig = px.scatter_3d(pca_result,
                    x='pca1', y='pca2', z='pca3',
                    color='music_genre')
background_color = 'rgb(230, 226, 218)'
dotsize = 1
fig.update_layout(scene=dict(
    xaxis=dict(
        backgroundcolor=background_color,
        gridcolor="white",
        showbackground=True,
        zerolinecolor="white", ),
    yaxis=dict(
        backgroundcolor=background_color,
        gridcolor="white",
        showbackground=True,
        zerolinecolor="white", ),
    zaxis=dict(
        backgroundcolor=background_color,
        gridcolor="white",
        showbackground=True,
        zerolinecolor="white", ), ),
    paper_bgcolor=background_color,
    plot_bgcolor=background_color,
    margin=dict(r=20, l=10, b=10, t=10)
    )
fig.update_traces(marker=dict(size=dotsize))
fig.show()

# to html
fig.write_html("pca.html")

# to sql
conn = sqlite3.connect('musicData.db')
c = conn.cursor()
pca_result.to_sql('pca_result', conn, if_exists='replace', index=False)
conn.commit()
conn.close()

print('Finish updating pca_result.')

# SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2,verbose=1)
tsne_result = tsne.fit_transform(X)
tsne_result = pd.DataFrame(tsne_result, columns=['tsne1', 'tsne2'])
tsne_result['music_genre'] = y
tsne_result['music_genre'] = tsne_result['music_genre'].astype('category')

# plot
fig = px.scatter(tsne_result,
                    x='tsne1', y='tsne2',
                    color='music_genre')
fig.show()

# to html
fig.write_html("tsne.html")


