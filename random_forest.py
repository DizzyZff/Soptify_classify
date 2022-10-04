import pandas as pd
import sqlite3

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV

#load from db
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

#random forest
"""random_grid = {'n_estimators': [100, 200, 300, 400, 500],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]}
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train)
print(rf_random.best_params_)"""



