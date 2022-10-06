import pandas as pd
import sqlite3
from sklearn.decomposition import PCA
import plotly.express as px
# load from db
data_path = 'musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
df = pd.read_sql_query("SELECT * FROM musicData_normal", conn)
conn.close()
print("Finish Loading")

corr = df.corr()
"""
fig1 = px.imshow(corr)

plt = px.imshow(corr, labels=dict(x="Features", y="Features", color="Correlation"),
                color_continuous_scale=px.colors.sequential.Cividis_r)
# colorscale = [[0, 'rgb(84, 84, 84)'], [1, 'rgb(230, 230, 218)']]
plt.update_layout(
    plot_bgcolor='rgb(234, 230, 222)',
    paper_bgcolor='rgb(234, 230, 222)',
    font_color='rgb(0, 0, 0)',
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    xaxis_zeroline=False,
    yaxis_zeroline=False,
    margin=dict(r=20, l=10, b=10, t=10)

plt.show()"""
#high quality image

print(df.head())
dropped_df = df.drop(['music_genre'], axis=1)
dropped_df = dropped_df.drop(['mode_Minor', 'mode_Major'], axis=1)
dropped_df = dropped_df.drop(['key'], axis=1)
dropped_df = dropped_df.drop(['popularity'], axis=1)
"""dropped_df = dropped_df.drop(['liveness'], axis=1)
dropped_df = dropped_df.drop(['tempo'], axis=1)
dropped_df = dropped_df.drop(['speechiness'], axis=1)
dropped_df = dropped_df.drop(['popularity'], axis=1)
dropped_df = dropped_df.drop(['valence'], axis=1)
dropped_df = dropped_df.drop(['danceability'], axis=1)"""
dropped_df = dropped_df.drop(['artist_name','track_name'], axis=1)
dropped_df = dropped_df.drop(['instance_id'], axis=1)
dropped_df = dropped_df.drop(['obtained_date'], axis=1)
new_df = dropped_df[['energy', 'loudness', 'acousticness']]
print(dropped_df.info())
# PCA
pca = PCA(n_components=2)
pca.fit(new_df)
pca_result = pca.transform(new_df)
print(pca.explained_variance_ratio_, "sum:", sum(pca.explained_variance_ratio_))
pca_result = pd.DataFrame(pca_result, columns=['pca1', 'pca2'])
df['pca1'] = pca_result['pca1']
df = df.drop(['energy', 'loudness', 'acousticness'], axis=1)
df = df.drop(['artist_name','track_name'], axis=1)
df = df.drop(['instance_id'], axis=1)
df = df.drop(['obtained_date'], axis=1)

conn = sqlite3.connect(data_path)
c = conn.cursor()
df.to_sql('musicData_pca', conn, if_exists='replace', index=False)
conn.close()
print("Finish Saving")




# plot


