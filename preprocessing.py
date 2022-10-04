import pandas as pd
import numpy as np
import sqlite3

#
data_path = 'musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
df = pd.read_sql_query("SELECT * FROM musicData", conn)
conn.close()
print("Finish Loading")

print(df.head(), df.info(), df.describe(),df.isnull().sum())
missing_count = df.isnull().sum()
value_count = df.isnull().count()
missing_rate = round(missing_count/value_count*100,2)
missing_df = pd.DataFrame({'missing_count':missing_count,'missing_rate':missing_rate})
print(missing_df)

barchart = missing_df.plot.bar(y ="missing_rate")
for missing_count,missing_rate in enumerate(missing_df['missing_rate']):
    barchart.text(missing_count,missing_rate,str(missing_rate)+'%',ha='center',va='bottom')

