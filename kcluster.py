import pandas as pd
import sqlite3

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.manifold import TSNE


# load from db
data_path = 'musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
df = pd.read_sql_query("SELECT * FROM musicData_normal", conn)
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
"""dropped_df = dropped_df.drop(['key'], axis=1)"""
"""dropped_df = dropped_df.drop(['liveness'], axis=1)
dropped_df = dropped_df.drop(['tempo'], axis=1)
dropped_df = dropped_df.drop(['speechiness'], axis=1)
dropped_df = dropped_df.drop(['popularity'], axis=1)
dropped_df = dropped_df.drop(['valence'], axis=1)
dropped_df = dropped_df.drop(['danceability'], axis=1)"""
dropped_df = dropped_df.drop(['artist_name','track_name'], axis=1)
dropped_df = dropped_df.drop(['instance_id'], axis=1)

#k-means clustering
km = KMeans(n_clusters=3)
km.fit(dropped_df)
y_kmeans = km.predict(dropped_df)
print(y_kmeans)
print(km.cluster_centers_)
print(km.inertia_)
print(km.n_iter_)
print(km.labels_)
print(km.n_clusters)

plt = px.scatter_3d(dropped_df, x='danceability', y='valence', z='popularity', color=y_kmeans)
plt.show()

