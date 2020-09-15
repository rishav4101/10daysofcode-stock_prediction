# Stock Prediction

"The field of study that gives computers the ability to learn without being explicitly programmed" is what Arthur Samuel described as Machine Learning.

**Machine Learning** or rather **ML** has found its applications in various fields in the past few years, some of which include Virtual Personal Assistants, Online Customer Support, Product Recommendations etc.


## So What exactly are we going to make?
In these 10 days of Code we will be using ML, taming the Stock market.
We'll create an application which will predict Stock Prices of a company using Supervised Machine Learning.

We will use libraries like `numpy`, `pandas`, `matplotlib`, `sklearn` and a few others. You may have heard about these libraries or you may be encountering their names for the first time and it may seem like jargon but don't worry, we will walkthrough all of them and make you familiar with them

### NumPy
Let's start with <code>NumPy</code>, Shall we? 
**NumPy**, which for Numerical Python is an Open Source python library used for working with arrays. It consists of multidimensional array objects and also has functions for working in domain of Linear Algebra, fourier transform and matrices. 

So Why use NumPy?

In Python we have lists that serve the purpose of arrays, but they are slow to process.

NumPy aims to provide an array object that is up to 50x faster that traditional Python lists.

The NumPy module does not come pre-installed with the python distribution and we have to install it seperately. It is often used with libraries such as scipy and matplotlib. 

Machine Learning data is universerally represented as arrays and in Python, data is almost universerally represented as NumPy arrays. To get started working on real data we need to know how to manipute and access the data correctly in NumPy arrays.

You can refer to the following tutorial which will demonstrate simple operations on our data such as :

- Converting list data to NumPy arrays
- Accessing data using pythonic indexing and slicing
- Resizing of the data 

[NumPy Arrays](https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/)

After going through the above tutorial, you now should be familiar with performing basic operations on Arrays.

You can also refer to the following resources to know further about NumPy
- [w3schools](https://www.w3schools.com/python/numpy_intro.asp)
- [NumPy documentation](https://numpy.org/doc/stable/)

So, lets continue and move on to our next topic **Pandas**

### Pandas
Pandas is a very popular open-source Python Library which provides high-performance data manipulation and analysis tools using its powerful data structures. The name Pandas is derived from the word Panel Data – an Econometrics from Multidimensional data.  Using Pandas, we can accomplish five typical steps in the processing and analysis of data, regardless of the origin of data — load, prepare, manipulate, model, and analyze.

Following are some of the key features of Pandas

- Fast and efficient DataFrame object with default and customized indexing.
- Tools for loading data into in-memory data objects from different file formats.
- Data alignment and integrated handling of missing data.
- Reshaping and pivoting of date sets.
- Label-based slicing, indexing and subsetting of large data sets.
- Columns from a data structure can be deleted or inserted.
- Group by data for aggregation and transformations.
- High performance merging and joining of data.
- Time Series functionality.

You can also refer to the following resources to know further about Pandas
- [tutorialspoint](https://www.tutorialspoint.com/python_pandas/index.htm)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

### Matplotlib
**Matplotlib** is an open source plotting library  which creates static , animated and interactive visualizations in python. It is largely used for the 2-dimensional plotting of arrays. It is a multi platform data visualization library which has been built upon NumPy arrays. One of its remarkable feature is that it lets the user interact with large chunk of data in very easily representable visuals. Matplotlib consists of several plots like line bar, pie chart, histogram, etc. Matplotlib comes with large number of plots ,which helps in determining trends and building correlation such as the line bar, scatter bar, histogram etc.

## A Short Tutorial
We have discussed in breif about some of the important libraries which are being used for data analysis. And So now, we would provide a very beginner friendly guide which will demonstrate how to use these libraries and will make you equipped so that you are able to tackle the actual project on your own.

Also, we will have to install these libraries into our system as they do not come pre-installed with python.

```
pip install numpy pandas matplotlib
```
You can create a new python file and follow along.


```
# importing the libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
```

<!-- In this Beginner friendly, step by step guide, we will now understand how to load a dataset and understand its structure with help of statistical summaries and data visualization. -->

We are going to use the `iris flowers` dataset. This dataset is famous because it is used as the “hello world” dataset in machine learning and statistics by pretty much everyone.The dataset contains 150 observations of iris flowers. There are four columns of measurements of the flowers in centimeters. The fifth column is the species of the flower observed. All observed flowers belong to one of three species.
We will load the iris data from **csv** file URL.

If you don't know what a csv file is, a comma-separated values (csv) file is a delimited text file that uses a comma to separate values. Each line of the file is a data record. Each record consists of one or more fields, separated by commas.

```
# Load Dataset

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
```

Now, lets take a look at the data we have loaded. Some of the ways to look at the data are as follows:

```
# Dimensions of the dataset
print(dataset.shape) 

# Peeking into the dataset
print(dataset.head(10)) # prints the first 10 rows of the dataset.

# Statistical Summary
print(dataset.describe()) 
# prints the summary of each column head(attribute)
# This helps in finding maximum, minimum, count, and even some percentiles.
```


Now we get a basic understanding about our data. Let's dive into the data visualization part.

We generally study two types of plots:
- **Univariate plots** to undersatnd each attribute (box and whisker plots, Histograms, etc)
- **Multivariate plots** to understand the relationship between varoius attributes (Scatter plots)

```
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
```
![Univariate plot](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2016/06/Box-and-Whisker-Plots-for-Each-Input-Variable-for-the-Iris-Flowers-Dataset-1024x768.png)


```
# histrograms
dataset.hist()
pyplot.show()
```
![Histogram of univariate plot](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2016/06/Histogram-Plots-for-Each-Input-Variable-for-the-Iris-Flowers-Dataset-1024x768.png)

```
#scatter plot matrix
scatter_matrix(dataset)
pyplot.show()
```
![Multivariate Plot](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2016/06/Scatter-Matrix-Plot-for-Each-Input-Variable-for-the-Iris-Flowers-Dataset-1024x768.png)

So now you must have an idea of what these libraries help us to do and you might have also got an intuition as to where you can use them in you projects. Although we have only discussed the topics in very brief and there a whole lot of things that these libraries offer and what we can do with them but this should be a really good checkpoint we have reached.

Now coming to our project, as we are dealing with the stock market and trying to predict stock prices the most important thing is **being able to Read Stocks**

## How to Read Stocks ? 
Reading stock charts, or stock quotes, is a crucial skill in being able to understand how a stock is performing, what is happening in the broader market and how that stock is projected to perform.

Stocks have **quote pages** or **charts**, which give both basic and more detailed information about the stock, its performance and the company on the whole. So, the next question that comes up is what makes up a stock chart? 

### Stock Charts
A Stock Chart is a set of information on a particular company's stock that generally shows information about price changes, current trading price, historical highs and lows, dividends, trading volume and other company financial information.

Also we would like to familiarise you some some basic terminologies of the stock market

#### Ticker Symbol
The ticker symbol is the symbol that is used on the stock exchange to delineate a given stock. For example, Apple's ticker is (AAPL) while Snapchat's ticker is (SNAP).

#### Open Price
The open price is simply the price at which the stock opened on any given day

#### Close Price
The close price is perhaps more significant than the open price for most stocks. The close is the price at which the stock stopped trading during normal trading hours (after-hours trading can impact the stock price as well). If a stock closes above the previous close, it is considered an upward movement for the stock. Vice versa, if a stock's close price is below the previous day's close, the stock is showing a downward movement.

## How to Read a Stock Chart ?
A Stock Chart is a little different than the basic information on the stock. Stock charts include plot lines which represent the price movements of the given stock. While the price lines are generally represented in a line or a mountain chart form, the price movements over a given period (generally six months to one year) is represented by thin lines.

### 1. The Price and Time Axes
Every stock chart has two axes - the price axis and the time axis. The horizontal (or bottom) axis shows the time period selected for the stock chart. The vertical (or side) axis shows the price of the stock. These two axes help plot the trend lines that represent the stock's price over time, and are the framework for the whole stock chart. 

![Price and Time axes](https://beginningstocktrader.com/wp-content/uploads/2016/02/Basic-Chart.jpg)

### 2. The Trend Line
This should be pretty obvious, but a good bit of the information one can glean from a stock chart can be found in the trend line. Depending on the type of chart one is looking at, one can choose different chart styles including the traditional line, mountain, bar, candlestick and other chart styles.
The  traditional **Line charts** simply track the price movements of a stock using the last price of that stock.

### 3. Trading Volume

In addition to just the trend of the stock's prices, the stock's trading volume is another key factor to look at when reading a stock chart.

The volume is generally indicated on the bottom of the stock chart in green and red bars (or sometimes blue or purple bars). The key thing to look out for when examining trading volume is spikes in trading volume, which can indicate the strength of a trend - whether it is high trading volume down or up. If a stock's price drops and the trading volume is high, it might mean that there is strength to the downward trend on the stock as opposed to a momentary blip (and vice versa if the price moves up)

### An Example Stock Chart
![Stock Chart](https://www.thestreet.com/.image/t_share/MTY3NTM5MzU3MjA0MDk2OTEw/image-placeholder-title.jpg)

So, this was little bit of knowledge about the Stock Market and how to read the Stock charts.

Now its time to get your hands dirty and begin setting up the project

## IDE/Editor Setup
Bhaiya pls check if we need to include any IDE or editor setup in this guide.

## Initializing our project

To work with Stock Market prices we first need to get reliable financial data. So where do we get our data from? `yfinance` solves our problem by offering a reliable, threaded, and Pythonic way to download historical market data from Yahoo! finance. And this library is installed pretty much the same way as we have installed other libraries. So go ahead and install it.

Also to download the stock prices data of a particular company using the `yfiance` library you will need to know their **Ticker Symbol**

[All stock ticker symbols](https://stockanalysis.com/stocks/)

![yfinance GitHub Repository](https://github.com/ranaroussi/yfinance)

Use the `yfinance` library to download the dataframe. The dataframe which we get contains daily data about the stock. The downloaded dataframe gives us lot of information including Opening Price, Closing Price, Volume, etc. But we are interested in the opening prices with their corresponding dates.

Also it would convinient to convert the dates to their corresponding time-stamps. And finally we will be having a dataframe which will contain our opening prices and time-stamps.

We are getting the day-wise and this much data is not sufficiently enough to train our model properly. So we use an `api` to get the stock prices of the previous day for every minute. Following is the link to the api endpoint

![Previous day Stock Prices](https://cloud.iexapis.com/stable/stock/aapl/chart/1d?token=sk_8a186cf264dc42d4963f5793b92ea911)

So now you get the data. You can use the `requests` and `json` modules to use the date and according append it to your dataframe.

Also to further improve the performance of our results and make even more accurate predictions we also add the current day's stock prices using the follwing api endpoint

![Today's Stock Prices](https://cloud.iexapis.com/stable/stock/aapl/intraday-prices/batch?token=sk_8a186cf264dc42d4963f5793b92ea911)


After appending the current day's prices we have sufficent number of records to train our models.

### Create a Validation Dataset

We need to know that the model we created is good. We are going to hold back some data that the algorithms will not get to see and we will use this data to get a second and independent idea of how accurate the best model might actually be.

We will split the loaded dataset into two, 80% of which we will use to train, evaluate and select among our models, and 20% that we will hold back as a validation dataset.

Following is a sample code

```
# Split-out validation dataset
dataset = prices_dataframe.values
X = ...
y = ...
# X and y are obtained by array slicing
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
```

The function `train_test_split()` comes from the `scikit-learn` library.

**scikit-learn** (also known as sklearn) is a free software machine learning library for the Python. Scikit-learn provides a range of supervised and unsupervised learning algorithms via a consistent interface in Python.
The library is focused on modeling data. It is not focused on loading, manipulating and summarizing data.

## Building our Models

We don’t know which algorithms would be good on this project or what configurations to use.

And So, we are testing with 6 different alogrithms:
- Linear Regression (LR)
- Lasso (LASSO)
- Elastic Net (EN)
- KNN (K-Nearest Neighnors)
- CART (Classification and Regression Trees)
- SVR (Support Vector Regression)

Let us briefly discuss about these algorithms

## Linear Regression
Linear regression is a supervised learning algorithm and tries to model the relationship between a continuous target variable and one or more independent variables by fitting a linear equation to the data.
For a linear regression to be a good choice, there needs to be a linear relation between independent variable(s) and target variable. There are many tools to explore the relationship among variables such as scatter plots and correlation matrix. 

## K-nearest neighbors (kNN) 
kNN is also a supervised learning algorithm that can be used to solve both classification and regression problems. The main idea behind kNN is that the value or class of a data point is determined by the data points around it.
kNN classifier determines the class of a data point by majority voting principle. For instance, if k is set to 15, the classes of 15 closest points are checked.

## Lasso
lasso (least absolute shrinkage and selection operator; also Lasso or LASSO) is a regression analysis method that performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the statistical model it produces.

## Elastic-Net 
The Elastic-Net is a regularised regression method that linearly combines both penalties i.e. L1 and L2 of the Lasso and Ridge regression methods. It is useful when there are multiple correlated features. The difference between Lasso and Elastic-Net lies in the fact that Lasso is likely to pick one of these features at random while elastic-net is likely to pick both at once.

```
# imports
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
```

```
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
```
The output of the above code gives us the accuracy estimations for each of our algorithms. We need to compare the models to each other and select the most accurate.

Once we are able to choose which results in the best accuracy, all we have to do is to
- Define the model
- Fit data into our model
- Make predictions

Plot your predictions alongwith the actual data and the two plots will nearly overlap.