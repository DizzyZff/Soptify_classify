import pandas as pd
import sqlite3
import plotly.express as px
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

data_path = 'musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
train = pd.read_sql_query("SELECT * FROM train", conn)
test = pd.read_sql_query("SELECT * FROM test", conn)
conn.close()
print("Finish Loading")



