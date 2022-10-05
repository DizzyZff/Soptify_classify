import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
from sklearn import metrics

# from db
from sklearn.naive_bayes import GaussianNB

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

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# predict
y_pred_prob = nb.predict_proba(X_test)
acc = metrics.accuracy_score(y_test, y_pred_prob[:, 1] > 0.5)
print("Accuracy: ", acc)

auc = metrics.roc_auc_score(y_test, y_pred_prob[:, 1])
print("AUC: ", auc)

# ROC
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob[:, 1])
fig = px.area(
    x=fpr, y=tpr,
    title='ROC Curve',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
)
fig.add_shape(
    type="line", line=dict(dash="dash"),
    x0=0, x1=1, y0=0, y1=1
)
fig.show()
