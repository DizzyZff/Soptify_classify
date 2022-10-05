import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


#load from db
data_path = 'musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
df_train = pd.read_sql_query("SELECT * FROM train", conn)
df_test = pd.read_sql_query("SELECT * FROM test", conn)
conn.close()
print("Finish Loading")

# train
X_train = df_train.drop(['music_genre'], axis=1)
y_train = df_train['music_genre']

# test
X_test = df_test.drop(['music_genre'], axis=1)
y_test = df_test['music_genre']

# find best parameters
