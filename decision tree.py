import pandas as pd
import sqlite3
import plotly.express as px
import torch
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

data_path = 'musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
train = pd.read_sql_query("SELECT * FROM train", conn)
test = pd.read_sql_query("SELECT * FROM test", conn)
conn.close()
print("Finish Loading")

X_train = train.drop(['music_genre'], axis=1)
y_train = train['music_genre']
y_train = y_train.astype('int')
X_test = test.drop(['music_genre'], axis=1)
y_test = test['music_genre']
y_test = y_test.astype('int')

# decision tree

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
score = metrics.accuracy_score(y_test, y_pred)
print("score: ", score)

# best
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=5, min_samples_split=2)
score_pre = cross_val_score(dtc, X_train, y_train, cv=10).mean()
print("score_pre: ", score_pre)

best_bucket ={
    'criterion': ['entropy', 'gini'],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 5, 10]
}

for i in range(10):
    dtc = DecisionTreeClassifier()
    random_search = RandomizedSearchCV(dtc, param_distributions=best_bucket, n_iter=10, cv=10, random_state=42)
    random_search.fit(X_train, y_train)
    print(random_search.best_params_)
    print(random_search.best_score_)
    score = cross_val_score(random_search, X_train, y_train, cv=10).mean()
    print("score: ", score)



