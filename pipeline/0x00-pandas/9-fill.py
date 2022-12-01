#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')


"""
Filling in the missing data points in the pd.DataFrame
"""
# The column Weighted_Price should be removed
df.pop("Weighted_Price")

# missing values in Close should be set to the previous row value
df['Close'] = df['Close'].ffill()

# missing values in High, Low, Open should be set to the same row’s Close value
df['Open'] = df['Open'].fillna(df['Close'])
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])

# missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
df.fillna({"Volume_(BTC)": 0, "Volume_(Currency)": 0}, inplace=True)


print(df.head())
print(df.tail())
