#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:08:51 2024

@author: sachin borade

"""

import pandas as pd 

# Loading the csv file 
df = pd.read_csv("Heart.csv")
df
# Removing the unwanted column 
df.drop(columns=['Unnamed: 0'],inplace=True)
df
# Seperating the numerical attributes from the above dataset 
numerical_data = df.select_dtypes(include=['number'])
numerical_data


# a) Find standard deviation, variance of every numerical attribute
std_dev = numerical_data.std()
print(std_dev)
variance = numerical_data.var()
print(variance)


# b) Find covariance and perform Correlation analysis using Correlation coefficient
covariance_matrix = numerical_data.cov()
print(covariance_matrix)
correlation_matrix = numerical_data.corr()
print(correlation_matrix)


# c) How many independent features are present in the given dataset?
# We can use the correlation matrix to identify independent features
# Independent features have correlation coefficients close to 0

independent_features = sum(correlation_matrix.abs().sum() < 1)
print(independent_features)


# d) Can we identify unwanted features?
# Unwanted features could be those with low correlation with the target variable 
# or with high correlation with other features
# You can set a threshold for correlation coefficients to identify such features
unwanted_features = correlation_matrix[correlation_matrix.abs() < 0.2].index.tolist()
print(unwanted_features)


# e) Perform the data discretization using equi frequency binning method on age
# attribute. We will use the 'cut' function in pandas to discretize the 'age' 
# attribute into bins of equal frequency
numerical_data['age_bins'] = pd.cut(numerical_data['Age'], bins=5)
numerical_data


# f) Normalize RestBP, chol, and MaxHR attributes using min-max normalization, 
# Z-score normalization, and decimal scaling normalization, Min-max normalization

numerical_data['RestBP_minmax'] = (numerical_data['RestBP'] - numerical_data['RestBP'].min()) / (numerical_data['RestBP'].max() - numerical_data['RestBP'].min())
numerical_data['chol_minmax'] = (numerical_data['Chol'] - numerical_data['Chol'].min()) / (numerical_data['Chol'].max() - numerical_data['Chol'].min())
numerical_data['MaxHR_minmax'] = (numerical_data['MaxHR'] - numerical_data['MaxHR'].min()) / (numerical_data['MaxHR'].max() - numerical_data['MaxHR'].min())
print(numerical_data)

# Z-score normalization
numerical_data['RestBP_zscore'] = (numerical_data['RestBP'] - numerical_data['RestBP'].mean()) / numerical_data['RestBP'].std()
numerical_data['chol_zscore'] = (numerical_data['Chol'] - numerical_data['Chol'].mean()) / numerical_data['Chol'].std()
numerical_data['MaxHR_zscore'] = (numerical_data['MaxHR'] - numerical_data['MaxHR'].mean()) / numerical_data['MaxHR'].std()
print(numerical_data)







