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
**Matplotlib** is an open source plotting library  which creates static , animated and interactive visualizations in python. It is largely used for the 2-dimensional plotting of arrays. It is a multi platform data visualization library which has been built upon NumPy arrays. One of its remarkable feature is that it lets the user interact with large chunk of data in very easily representable visuals. Matplotlib consists of several plots like line bar, pie chart, histogram, etc.
Matplotlib comes with large number of plots ,which helps in determining trends and building correlation such as the line bar, scatter bar, histogram etc.

