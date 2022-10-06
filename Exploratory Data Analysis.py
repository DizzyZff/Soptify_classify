import pandas as pd
import sqlite3
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'axes.facecolor':'#e6e2da', 'figure.facecolor':'#e6e2da'})
# load from db

data_path = 'musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
df = pd.read_sql_query("SELECT * FROM musicData_clean", conn)
conn.close()
print("Finish Loading")

cat_df = df[['mode_Major','mode_Minor', 'music_genre', 'key']]
num_df = df.drop(['mode_Major','mode_Minor', 'music_genre','key'], axis=1)
print(cat_df.head(), num_df.head())

# pairplot
"""background_color = "#e6e2da"
fig = sns.pairplot(num_df, plot_kws={'alpha': 0.6, 's': 3, 'edgecolor': '#545454'})
fig.fig.set_facecolor(background_color)
plt.show()
fig.savefig('pairplot.png')"""

conn = sqlite3.connect(data_path)
c = conn.cursor()
df_pca = pd.read_sql_query("SELECT * FROM musicData_pca", conn)
conn.close()
print("Finish Loading")

"""cat_df = df_pca[['mode_Major','mode_Minor', 'music_genre', 'key']]"""
num_df = df_pca.drop(['music_genre','key'], axis=1)
print(cat_df.head(), num_df.head())

# pairplot
fig = sns.pairplot(num_df, plot_kws={'alpha': 0.6, 's': 3, 'edgecolor': '#545454'})
plt.show()
fig.savefig('pairplot_pca.png')

# corr
corr = num_df.corr()
fig1 = px.imshow(corr)
fig1.show()