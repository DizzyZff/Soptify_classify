import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

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

# knn
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# predict
y_pred = knn.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

class_combinations = []
for i in range(0, 10):
    for j in range(i+1, 10):
        class_combinations.append((i, j))
        class_combinations.append((j, i))

print(class_combinations)
roc_auc_ovo = {}
for i in range(0, len(class_combinations)):
    comb = class_combinations[i]
    c1 = comb[0]
    c2 = comb[1]
    c1_index = y_test