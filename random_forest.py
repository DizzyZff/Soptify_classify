import numpy as np
import pandas as pd
import sqlite3
import plotly.express as px
import preprocessing as pp

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

#load from db
from sklearn.preprocessing import StandardScaler

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

# Feature Scaling
"""scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = RandomForestClassifier(n_estimators=400, random_state=0, max_depth=10, min_samples_split=2, min_samples_leaf=5, max_features='auto', bootstrap=True)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# reverse factorize
reversefactor = dict(zip(range(10), pp.definitions))
print(reversefactor)

print(y_pred, y_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy' + str(accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))"""

rfc = RandomForestClassifier(n_estimators=100, random_state=90)
score_pre = cross_val_score(rfc, X_train, y_train, cv=10).mean()
print("score_pre: ", score_pre)

score1 = []
for i in range(0, 200, 10):
    rfc = RandomForestClassifier(n_estimators=i + 1, n_jobs=-1, random_state=90)
    score = cross_val_score(rfc, X_train, y_train, cv=10).mean()
    score1.append(score)
print(max(score1), (score1.index(max(score1)) * 10 + 1))
fig = px.line(x=np.arange(1, 201, 10), y=score1)
fig.show()







