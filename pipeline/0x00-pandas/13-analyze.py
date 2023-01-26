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


stat = df1.drop(columns=['timestamp']).describe()
print(stat)
