import pandas as pd
from datetime import datetime, timedelta as td

pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)

data = pd.read_csv('data/_ts.csv')
# data = pd.read_csv('data/_ts_partial.csv')


start_date = datetime(2016, 3, 31)

data['dt'] = data.apply(lambda row: start_date + td(days=row['day']) + td(minutes=int(row['interval']) * 10), axis=1)

to_drop = ['datetime', 'date']

for i in range(144):
    to_drop.append('int_' + str(i))

for i in range(1, 31):
    to_drop.append('p_' + str(i))

data['weekday'] = 0
for i in range(7):
    data['wd_' + str(i)] = data['wd_' + str(i)]
    data['weekday'] = data['weekday'] + i * data['wd_' + str(i)]
    to_drop.append('wd_' + str(i))

data.drop(to_drop, axis=1, inplace=True)

print(data[data['day'] == 8])
print(data.columns)

data.to_csv('data/_ts_pretty.csv')

# data.head(1000).to_csv('data/_ts_partial.csv', index=False)
