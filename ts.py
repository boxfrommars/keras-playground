import pandas as pd

pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)

data = pd.read_csv('data/_ts_pretty.csv')

print(data)
