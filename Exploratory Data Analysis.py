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

cat_df = df['music_genre', 'key', "mode_Major", "mode_Minor"]
num_df = df.drop(['music_genre', 'key', "mode_Major", "mode_Minor"], axis=1)
print(cat_df.head(), num_df.head())

fig2 = plt.figure(figsize=(30, 30))
sns.pairplot(num_df, height=5)
sns.set(font_scale=2)
plt.tight_layout()
plt.show()
#high quality image
fig2.savefig('pairplot.png', dpi=300)

fig, ax = plt.subplots(len(cat_df.columns), len(cat_df.columns), figsize=(50, 50))
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

"""# categorical vs numerical/ first 5 columns
fig3, ax = plt.subplots(len(cat_df.columns), 5, figsize=(50, 50))
for i in range(0, len(cat_df.columns)):
    cat = cat_df.columns[i]
    for j in range(0, 5):
        num = num_df.columns[j]
        chart = sns.boxplot(x=cat, y=num, data=df, ax=ax[i][j])
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

plt.show()
# high quality image
fig3.savefig('cat_num1.png', dpi=300)

# categorical vs numerical/ next 5 columns
fig4, ax = plt.subplots(len(cat_df.columns), 5, figsize=(50, 50))
for i in range(0, len(cat_df.columns)):
    cat = cat_df.columns[i]
    for j in range(0, 5):
        num = num_df.columns[j+5]
        chart = sns.boxplot(x=cat, y=num, data=df, ax=ax[i][j])
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

plt.show()
# high quality image
fig4.savefig('cat_num2.png', dpi=300)

# categorical vs numerical/ last 3 columns
fig5, ax = plt.subplots(len(cat_df.columns), 3, figsize=(50, 50))
for i in range(0, len(cat_df.columns)):
    cat = cat_df.columns[i]
    for j in range(0, 3):
        num = num_df.columns[j+10]
        chart = sns.boxplot(x=cat, y=num, data=df, ax=ax[i][j])
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

plt.show()
# high quality image
fig5.savefig('cat_num3.png', dpi=300)"""


