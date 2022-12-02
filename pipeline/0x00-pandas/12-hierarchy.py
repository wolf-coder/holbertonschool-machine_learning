#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

"""
- Based on 11-concat.py, rearrange the MultiIndex levels such that
timestamp is the first level:
  + Concatenate the bitstamp and coinbase tables from timestamps
1417411980 to 1417417980, inclusive
  + Add keys to the data labeled bitstamp and coinbase respectively
  + Display the rows in chronological order
"""


df1 = df1.set_index('Timestamp')
df1 = df1.loc[1417411980:1417417980]
df2 = df2.set_index('Timestamp')
df2 = df2.loc[1417411980:1417417980]

df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
df = df.reorder_levels([1, 0]).sort_values(by=['Timestamp'])


print(df)
