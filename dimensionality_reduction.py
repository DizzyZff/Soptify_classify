import pandas as pd
import sqlite3
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.manifold import TSNE


# load from db
data_path = 'musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
df = pd.read_sql_query("SELECT * FROM musicData_clean", conn)
conn.close()
print("Finish Loading")

corr = df.corr()
fig1 = px.imshow(corr)

plt = px.imshow(corr, labels=dict(x="Features", y="Features", color="Correlation"),
                color_continuous_scale=px.colors.sequential.Cividis_r)
# colorscale = [[0, 'rgb(84, 84, 84)'], [1, 'rgb(230, 230, 218)']]
plt.update_layout(
    plot_bgcolor='rgb(230, 226, 218)',
    paper_bgcolor='rgb(230, 226, 218)',
    font_color='rgb(0, 0, 0)',
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    xaxis_zeroline=False,
    yaxis_zeroline=False,
    margin=dict(r=20, l=10, b=10, t=10)
)
plt.show()
plt.write_html("corr.html")

print(df.head())
dropped_df = df.drop(['music_genre'], axis=1)
dropped_df = dropped_df.drop(['mode_Minor', 'mode_Major'], axis=1)
# PCA
pca = PCA(n_components=3, whiten=True)
pca.fit(dropped_df)
pca_result = pca.transform(dropped_df)
pca_result = pd.DataFrame(pca_result, columns=['pca1', 'pca2', 'pca3'])
pca_result['music_genre'] = df['music_genre']
pca_result['mode_Major'] = df['mode_Major']
pca_result['mode_Minor'] = df['mode_Minor']
pca_result['music_genre'] = pca_result['music_genre'].astype('category')

# plot
fig = px.scatter_3d(pca_result, x='pca1', y='pca2', z='pca3', color='music_genre')
background_color = 'rgb(230, 226, 218)'
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
    margin=dict(r=20, l=10, b=10, t=10))
fig.update_traces(marker=dict(size=1))
fig.show()
fig.to_html("pca.html")
# to sql
conn = sqlite3.connect('musicData.db')
c = conn.cursor()
pca_result.to_sql('pca_result', conn, if_exists='replace', index=False)
conn.commit()
conn.close()

print('Finish updating pca_result datasets.')
"""
# t sne
tsne = TSNE(n_components=3, verbose=1, perplexity=10)
tsne_result = tsne.fit_transform(dropped_df)
tsne_result = pd.DataFrame(tsne_result, columns=['tsne1', 'tsne2', 'tsne3'])
tsne_result['music_genre'] = df['music_genre']
score = tsne.kl_divergence_

# plot
fig = px.scatter_3d(tsne_result, x='tsne1', y='tsne2', z='tsne3', color='music_genre')
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
    margin=dict(r=20, l=10, b=10, t=10))
fig.show()

# to sql
conn = sqlite3.connect('musicData.db')
c = conn.cursor()
tsne_result.to_sql('tsne_result', conn, if_exists='replace', index=False)
conn.commit()
conn.close()
"""