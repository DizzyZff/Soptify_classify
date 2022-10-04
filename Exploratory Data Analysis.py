import pandas as pd
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# load from db
data_path = 'musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
df = pd.read_sql_query("SELECT * FROM musicData_clean", conn)
conn.close()
print("Finish Loading")

cat_df = df[['mode', 'music_genre', 'key']]
num_df = df.drop(['mode', 'music_genre', 'key'], axis=1)
print(cat_df.head(), num_df.head())

corr = num_df.corr()
fig1 = plt.figure(figsize=(15, 15))
sns.heatmap(corr, annot=True, cmap='GnBu', linewidths=0.5)
sns.set(font_scale=2)
plt.show()
fig1.savefig('heatmap.png', dpi=300)

# TODO: figure out how to save the figure
fig2 = plt.figure(figsize=(30, 30))
sns.pairplot(num_df, height=5)
sns.set(font_scale=2)
plt.tight_layout()
plt.show()
#high quality image
fig2.savefig('pairplot.png', dpi=300)


fig, ax = plt.subplots(len(cat_df.columns), len(cat_df.columns), figsize=(30, 30))
for i in range(0, len(cat_df.columns)):
    primary = cat_df.columns[i]
    for j in range(0, len(cat_df.columns)):
        secondary = cat_df.columns[j]
        if i != j:
            chart = sns.countplot(x=primary, hue=secondary, data=cat_df, ax=ax[i][j])
            chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.show()
# high quality image
fig.savefig('cat_df.png', dpi=300)
