import pandas as pd
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.api.types import is_numeric_dtype, is_string_dtype

#load from db
data_path = 'musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
df = pd.read_sql_query("SELECT * FROM musicData", conn)
conn.close()
print("Finish Loading")

print(df.head())
print(df.info())



