import sqlite3
import pandas as pd
import sklearn
import torch

#import model
model = torch.load('model.pt')

#load test data
data_path = 'musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
test = pd.read_sql_query("SELECT * FROM test", conn)
conn.close()

#test
test_x = test.drop(['music_genre'], axis=1)
test_y = test['music_genre']

#predict
pred = model.predict(test_x)
auc = sklearn.metrics.roc_auc_score(test_y, pred)
print(auc)

