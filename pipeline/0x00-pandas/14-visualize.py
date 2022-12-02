#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE
# The column Weighted_Price should be removed
df.pop("Weighted_Price")

# Rename the column Timestamp to Date
df = df.rename(columns={"Timestamp": "Date"})

# Convert the timestamp values to date values
df['Date'] = pd.to_datetime(
    df["Date"],
    unit='s'
)

# Index the data frame on Date
df = df.set_index("Date")
# missing values in Close should be set to the previous row value
df['Close'] = df['Close'].ffill()

# missing values in High, Low, Open should be set to the same row’s Close value
df['Open'] = df['Open'].fillna(df['Close'])
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])

# missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
df.fillna({"Volume_(BTC)": 0, "Volume_(Currency)": 0}, inplace=True)

"""
Plot the data from 2017 and beyond at daily intervals and
group the values of the same day such that:
"""
df = df.loc['2017-01-01 00:00:00':]

# High: max
High = df['High'].groupby(pd.Grouper(freq='D')).max()
# Low: min
Low = df['Low'].groupby(pd.Grouper(freq='D')).min()

# Open: mean
Open = df['Open'].groupby(pd.Grouper(freq='D')).mean()

# Close: mean
Close = df['Close'].groupby(pd.Grouper(freq='D')).mean()

# Volume(BTC): sum
volume_btc = df['Volume_(BTC)'].groupby(pd.Grouper(freq='D')).sum()

# Volume(Currency): sum
volume_currency = df['Volume_(Currency)'].groupby(pd.Grouper(freq='D')).sum()

plt.figure(figsize=(16, 8))
plt.plot(Open, label='Open')
plt.plot(High, label='High')
plt.plot(Low, label='Low')
plt.plot(Close, label='Close')
plt.plot(volume_btc, label='Volume_(BTC)')
plt.plot(volume_currency, label='Volume_(Currency)')
plt.legend()
plt.show()
