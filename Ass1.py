#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 21:53:27 2024

@author: sachin
"""

# Heart Disease Dataset

#Performing the following operation on the dataset 
#a) Find Missing Values and replace the missing values with suitable alternative.
#b) Remove inconsistency (if any) in the dataset.
#c) Prepare boxplot analysis for each numerical attribute. Find outliers (if any) in each attribute in the dataset.
#d) Draw histogram for any two suitable attributes (E.g. age and Chol attributes for above dataset)
#e) Find data type of each column.
#f) Finding out Zero's.
#g) Find Mean age of patients considering above dataset.
#h) Find shape of data. 


# a) Find Missing Values and replace the missing values with suitable alternative.

# Importing the Module
import pandas as pd

# Reading the csv file
df = pd.read_csv("/home/sachin/Desktop/DMW/Heart.csv")
df

# Finding datatype for each dataset
df.dtypes

# Checking some columns and rows of the dataset
df.head()

# Checking the shape of dataset
df.shape

# Checking the NULL values
df.isnull()

# Checking the null values in each attribute
df.isnull().sum()

#Filling NULL values with value = 5
df1 = df.fillna(value = 5)
df1.isnull().sum()
# In df1 we replaced the null values with 5.

#Filling the null values with previous values
df2 = df.fillna(method='pad')
df2
# In df2 we replaced the null values with previous value from given column

#Filling the null values with next values
df3 = df.fillna(method='bfill')
df3
# In df3 we replaced the null values with next value from the column 

#Filling null values with next columns
df4 = df.fillna(method='bfill',axis=1)
df4
# In df4 we replaced the null values with adjacent next column values

#Filling null values with previous columns
df5 = df.fillna(method='pad',axis=1)
df5
# In df5 we replacd the null values with adjacent previous column values

#Filling null values with mean values
df6 = df.fillna(value=df['Ca'].mean())
df6
# The best way to fill the NULL values with Mean value

#Filling null values with minimum values
df7 = df.fillna(value=df['Ca'].min())
df7
# Here we are filling the NULL values with lowest value from the given column


#Filling data with maximum values
df8 = df.fillna(value=df['Ca'].max())
df8
# Here we are filling the NULL values with Maximum value from column

# Dropping the NULL values
df9 = df.dropna(how='all')
df9
# how = 'all' It only remove rows where all values are NaN.

# Dropping NULL values
df10 = df.dropna(how='any')
df10
# how ='any' It will remove rows which having single NaN values




# b) Remove inconsistency (if any) in the dataset.

# If any duplicates in the dataset then remove the dataset
df = df.drop_duplicates()
df 
# No duplicates in the dataset



# c) Prepare boxplot analysis for each numerical attribute. Find outliers 
# (if any) in each attribute in the dataset.
  
import seaborn as sns
import matplotlib.pyplot as plt  

df.dtypes

# Taking only numerical columns for plotting the box plot
numerical_col = df.select_dtypes(include='number').columns
numerical_col

# Plotting boxplot for each column
for col in numerical_col:
    plt.figure(figsize=(10,6))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot for {col}")
    plt.show()
# As per the observation we can see outlier in 
# 1) RestBP 
# 2) Chol
# 3) Fbs
# 4) MaxHR
# 5) Oldpeak
# 6) Ca

# d) Draw histogram for any two suitable attributes 
# (E.g. age and Chol attributes for above dataset)
df.columns

#plotting the histogram
# Chest pain
chest_pain = df['ChestPain']
# Blood pressure
blood_pressure = df['RestBP']

#size
plt.figure(figsize=(10,6))

plt.hist(chest_pain,bins=20,color='red')
plt.title("Histogram of Chest Pain")
plt.xlabel("Chest Pain")
plt.ylabel("Frequency")


#size
plt.figure(figsize=(10,6))

plt.hist(blood_pressure,bins=20,color='blue')
plt.title("Histogram of Blood Pressure")
plt.xlabel("Blood Pressure")
plt.ylabel("Frequency")


# e) Find data type of each column.

df.dtypes



# f) Finding out Zero's.

# For these we are using only numerical dataset
numerical_col

# Zeros in Age column
zeros_in_age = (df['Age'] == 0).sum()
print(f'Total zeros in the "Age" column: {zeros_in_age}')

# we dont want to find zeros in Sex column as 0 and 1 Represets Female/Male

# Zeros in RestBP column
zeros_in_RestBP = (df['RestBP'] == 0).sum()
print(f'Total zeros in the "RestBP" column: {zeros_in_RestBP}')

#Zeros in Chol Column
zeros_in_chol = (df['Chol'] == 0).sum()
print(f'Total zeros in the "Chol" column: {zeros_in_chol}')

#Zeros in Chol Column
zeros_in_fbs = (df['Fbs'] == 0).sum()
print(f'Total zeros in the "Fbs" column: {zeros_in_fbs}')

#Zeros in RestECG Column
zeros_in_ecg = (df['RestECG'] == 0).sum()
print(f'Total zeros in the "RestECG" column: {zeros_in_ecg}')

#Zeros in maxHR COlumn
zeros_in_maxHR = (df['MaxHR'] == 0).sum()
print(f'Total zeros in the "MaxHR" column: {zeros_in_maxHR}')

#Zeros in ExAng
zeros_in_exang = (df['ExAng'] == 0).sum()
print(f'Total zeros in the "ExAng" column: {zeros_in_exang}')

#Zeros in Oldpeak COlumn
zeros_in_oldpeak = (df['Oldpeak'] == 0).sum()
print(f'Total zeros in the "Oldpeak" column: {zeros_in_oldpeak}')

#Zeros in slope column
zeros_in_slope = (df['Slope'] == 0).sum()
print(f'Total zeros in the "Slope" column: {zeros_in_slope}')

#Zeros in Ca Column
zeros_in_ca = (df['Ca'] == 0).sum()
print(f'Total zeros in the "Ca" column: {zeros_in_ca}')


# g) Find Mean age of patients considering above dataset.

df['Age'].mean()
# The mean of Patient is 54 years.



# h) Find shape of data.

df.shape
