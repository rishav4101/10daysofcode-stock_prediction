#all imports
# import urllib3
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
import time 
import json
import datetime
import requests

# yfinance dataframe
yf.pdr_override() 

#set company ticker symbol
company = "AAPL"

# download dataframe
# SET YOUR ANALYSIS DATE WINDOW --> HERE SET TO YEAR 
data = pdr.get_data_yahoo(company, start="2020-01-01", end="2020-09-10")

# dataframe for model
prices = data[data.columns[0:1]]
prices.reset_index(level=0,inplace = True)

# converting date to timestamps
prices["timestamp"] = (pd.to_datetime(prices.Date).astype(int)) // (10**9)

# dropping date column
prices = prices.drop(['Date'], axis=1)

# convert to lists

# This is a list
last_day_prices = json.loads((requests.get('https://cloud.iexapis.com/stable/stock/aapl/chart/1d?token=sk_8a186cf264dc42d4963f5793b92ea911')).content) 

for price in last_day_prices:
    #price.date
    #price.time
    #price.open

    #single timestamp

    #to dataframe timestamp or open

this_day_prices = json.loads((requests.get("https://cloud.iexapis.com/stable/stock/aapl/intraday-prices/batch?token=sk_8a186cf264dc42d4963f5793b92ea911")).content)

for price in this_day_prices:
    #price.date
    #price.time
    #price.open

    #single timestamp

    #to dataframe timestamp or open

