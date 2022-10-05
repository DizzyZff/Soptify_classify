import numpy as np
import pandas as pd
import sqlite3
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

data_path = 'musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
train = pd.read_sql_query("SELECT * FROM train", conn)
test = pd.read_sql_query("SELECT * FROM test", conn)
conn.close()
print("Finish Loading")

#train
X_train = train.drop(['music_genre'], axis=1)
y_train = train['music_genre']
X_test = test.drop(['music_genre'], axis=1)
y_test = test['music_genre']

# gradient boosting
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0)
score_pre = cross_val_score(gbc, X_train, y_train, cv=10).mean()
print("score_pre: ", score_pre)
