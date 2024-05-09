# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:01:36 2024

@author: Admin
"""
'''
Problem Statements:
    1. Use any suitable dataset (e.g. https://www.kaggle.com/zhaoyingzhu/heartesv) and Perform
following operation on given dataset suitable programming language,
a) Find Missing Values and replace the missing values with suitable alternative.
b) Remove inconsistency (if any) in the dataset.
c) Prepare boxplot analysis for each numerical attribute. Find outliers (if any) in each attribute in
the dataset.
d) Draw histogram for any two suitable attributes (E.g. age and Chol attributes for above dataset)
e) Find data type of each column.
f) Finding out Zero's.
g) Find Mean age of patients considering above dataset.
h) Find shape of data.
'''

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mstats

# Step a: Load the dataset
data = pd.read_csv('C:/Users/Admin/Downloads/Heart (1).csv')

# Step a: Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:")
print(missing_values)

# Step a: Replace missing values with suitable alternatives (mean for numerical columns)
# Exclude non-numeric columns from mean calculation
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Step a: Replace missing values with suitable alternatives (mean for numerical columns)
#data.fillna(data.mean(), inplace=True)

# Step b: Check for incorrect data types
incorrect_data_types = data.dtypes[data.dtypes == 'object']
print("Columns with incorrect data types:")
print(incorrect_data_types)

# Step c: Prepare boxplot analysis for each numerical attribute
numerical_attributes = data.select_dtypes(include=['int64', 'float64']).columns

for column in numerical_attributes:
    # Create a boxplot for the current numerical column
    plt.figure(figsize=(8, 6))
    data.boxplot(column=[column])
    plt.title('Boxplot of ' + column)
    plt.show()

# Step c: Identify outliers using IQR method
for column in numerical_attributes:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR))).sum()
    print("Outliers in", column, ":", outliers)

# Step d: Draw histograms for two numerical attributes
data[['Age', 'Chol']].hist(figsize=(10, 6))
plt.suptitle('Histograms of Age and Chol')
plt.show()

# Step e: Find data type of each column
data_types = data.dtypes
print("Data Types:")
print(data_types)

# Step f: Count zeros in numerical columns
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
zero_counts = (data[numerical_columns] == 0).sum()
print("Counts of Zeros in Numerical Columns:")
print(zero_counts)

# Step g: Find mean age of patients
mean_age = data['Age'].mean()
print("Mean Age of Patients:", mean_age)

# Step h: Find shape of data
shape = data.shape
print("Shape of Data:", shape)

