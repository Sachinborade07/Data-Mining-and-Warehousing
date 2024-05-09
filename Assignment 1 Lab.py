# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 18:17:10 2024

@author: Gaurav Bombale

Assignment 1 - Data Mining Lab
Heart Csv file
"""


import pandas as pd
import seaborn as sns


df=pd.read_csv("D:/Data_Mining/Heart.csv")

print(df.head(10))

df.columns


df.isnull()

df.isnull().sum()


# Ca , Thal has null values, treat the null values with mean
df["Ca"].mean()
df.Ca.fillna(df["Ca"].mean(), inplace=True)
df.isnull().sum()

df.Thal.fillna(df.Thal.mode(), inplace=True)
Thal_mode = df["Thal"].mode()[0]

df["Thal"].fillna(Thal_mode, inplace=True)

df.isnull().sum()
'''
Unnamed: 0    0
Age           0
Sex           0
ChestPain     0
RestBP        0
Chol          0
Fbs           0
RestECG       0
MaxHR         0
ExAng         0
Oldpeak       0
Slope         0
Ca            0
Thal          0
AHD           0
dtype: int64
'''

# finding duplicate 
duplicate=df.duplicated()

duplicate

sum(duplicate)

df.columns
# Box plot on Age column
sns.boxplot(df.Age)
# No outliers on Age column

sns.boxplot(df.RestBP)
# Some Outliers on RestBP column

sns.boxplot(df.Chol)
# Some outliers on Chol column

sns.boxplot(df.MaxHR)
# One outlier on MaxHR column

sns.boxplot(df.Oldpeak)
# Outlier is found on Oldpeak column

sns.boxplot(df.Slope)
# No outlier found on Slope column

sns.boxplot(df.Ca)
# One outlier on Ca column

# Boxplot on whole dataframe
sns.boxplot(df)
# Many columns having outliers

# 4- Draw histogram for any two suitable attributes 
# (E.g. age and Chol attributes for above dataset)
sns.histplot(data=df, x="Age", y="Chol", hue="Sex")

# hue as "Fbs" column
sns.histplot(data=df, x="Age", y="Chol", hue="Fbs")

# 5- Find data type of each column.
df.dtypes
'''
Unnamed: 0      int64
Age             int64
Sex             int64
ChestPain      object
RestBP          int64
Chol            int64
Fbs             int64
RestECG         int64
MaxHR           int64
ExAng           int64
Oldpeak       float64
Slope           int64
Ca            float64
Thal           object
AHD            object
dtype: object
'''

# Finding out zeros
z=(df == 0).sum()
print("Number of zeros : ",z)

# 7- Find Mean age of patients considering above dataset.
# Mean age of patientes
round(df.Age.mean())
# 54 is the mean age of the patients

# 8- Find shape of data.
df.shape