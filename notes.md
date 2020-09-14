# Stock Prediction

"The field of study that gives computers the ability to learn without being explicitly programmed" is what Arthur Samuel described Machine Learning.

**Machine Learning** or rather **ML** has found its applications in various fields in the past few years, some of which include Virtual Personal Assistants, Online Customer Support, Product Recommendations etc.


## So What exactly are we going to make?
In these 10 days of Code we will be using ML, taming the Stock market.
We'll create an application which will predict Stock Prices of a company using **LTSM**(Long Term Short Memory Model). (Rishav thora aur explain kar dena yeh wala part)

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
#prints the summary of each column head(attribute).This helps in finding maximum, minimum, count, and even some percentiles.
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
