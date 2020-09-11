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
from numpy import arange
from numpy import set_printoptions
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt 

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
    datetime_str = price['date'] + " " + price['minute']
    datetime_object = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
    timestamp = datetime.datetime.timestamp(datetime_object)
    open_price = price['open']
    df = pd.DataFrame([[open_price, int(timestamp)]], columns=['Open', 'timestamp'])
    prices = prices.append(df, ignore_index=True)


this_day_prices = json.loads((requests.get("https://cloud.iexapis.com/stable/stock/aapl/intraday-prices/batch?token=sk_8a186cf264dc42d4963f5793b92ea911")).content)


for price in this_day_prices:
    datetime_str = price['date'] + " " + price['minute']
    datetime_object = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
    timestamp = datetime.datetime.timestamp(datetime_object)
    open_price = price['open']
    df = pd.DataFrame([[open_price, int(timestamp)]], columns=['Open', 'timestamp'])
    prices = prices.append(df, ignore_index=True)


dataset = prices.values #array

X = dataset[:,1].reshape(-1,1)
Y = dataset[:,0:1]

validation_size = 0.15
seed = 7

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# set_printoptions(precision=3)


# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = "r2"

# Spot-Check Algorithms
models = []
models.append((' LR ', LinearRegression()))
models.append((' LASSO ', Lasso()))
models.append((' EN ', ElasticNet()))
models.append((' KNN ', KNeighborsRegressor()))
models.append((' CART ', DecisionTreeRegressor()))
models.append((' SVR ', SVR()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    # print(cv_results)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Define model
model = DecisionTreeRegressor()
# Fit to model
model.fit(X_train, Y_train)
# predict
predictions = model.predict(X)
print(mean_squared_error(Y, predictions))

# %matplotlib inline 

fig= plt.figure(figsize=(12,6))

plt.plot(X,Y)
plt.plot(X,predictions)
plt.show()