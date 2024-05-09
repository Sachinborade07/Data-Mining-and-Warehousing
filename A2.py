# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:19:23 2024

@author: Admin
"""

'''
Problem Statements:
    2. Data Discretization and Data Normalization. Use any suitable dataset (e.g. heart dataset
https://www.kaggle.com/zhaoyingzhu/heartesv ). Perform following operations on given dataset
suitable programming language.
a) Find standard deviation, variance of every numerical attribute.
b) Find covariance and perform Correlation analysis using Correlation coefficient.
c) How many independent features are present in the given dataset?
d) Can we identify unwanted features?
e) Perform the data discretization using equi frequency binning method on age attribute
f) Normalize RestBP, chol, and MaxHR attributes (considering above dataset) using min-max
normalization, Z-score normalization, and decimal scaling normalization.
'''

import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('C:/Users/Admin/Downloads/Heart (1).csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# a) Find standard deviation and variance of every numerical attribute
# Select only numerical columns
numerical_attributes = data.select_dtypes(include=['int64', 'float64'])

# Calculate standard deviation and variance for each numerical attribute
std_deviation = numerical_attributes.std()
variance = numerical_attributes.var()

print("\nStandard Deviation of Numerical Attributes:")
print(std_deviation)
print("\nVariance of Numerical Attributes:")
print(variance)

# b) Find covariance and perform Correlation analysis using Correlation coefficient
# Calculate covariance matrix
covariance_matrix = numerical_attributes.cov()

# Perform Correlation analysis using Correlation coefficient (Pearson correlation coefficient)
correlation_matrix = numerical_attributes.corr()

print("\nCovariance Matrix:")
print(covariance_matrix)
print("\nCorrelation Matrix:")
print(correlation_matrix)

# c) How many independent features are present in the given dataset?
# Calculate the number of independent features using the correlation matrix
# Independent features are those with correlation coefficient close to 0 (or absolute value less than a threshold)
threshold = 0.1  # Set a threshold for correlation coefficient
independent_features = ((correlation_matrix.abs() < threshold) & (correlation_matrix.abs() != 1)).sum().sum()

print("\nNumber of independent features:", independent_features)

# d) Can we identify unwanted features?
# Unwanted features can be identified based on low variance or high correlation with other features
# Let's identify features with low variance
low_variance_features = variance[variance < 0.1]  # Select features with variance less than a threshold
print("\nUnwanted features (low variance):")
print(low_variance_features)

# e) Perform the data discretization using equi frequency binning method on age attribute
# Discretize the 'Age' attribute using equi frequency binning method
num_bins = 5  # Number of bins
labels = range(1, num_bins + 1)
data['Age_Binned'] = pd.qcut(data['Age'], q=num_bins, labels=labels)

print("\nDiscretized Age attribute:")
print(data[['Age', 'Age_Binned']].head())

# f) Normalize RestBP, chol, and MaxHR attributes using different normalization techniques

# Min-Max Normalization
data['RestBP_MinMax'] = (data['RestBP'] - data['RestBP'].min()) / (data['RestBP'].max() - data['RestBP'].min())
data['Chol_MinMax'] = (data['Chol'] - data['Chol'].min()) / (data['Chol'].max() - data['Chol'].min())
data['MaxHR_MinMax'] = (data['MaxHR'] - data['MaxHR'].min()) / (data['MaxHR'].max() - data['MaxHR'].min())

# Z-Score Normalization
data['RestBP_ZScore'] = (data['RestBP'] - data['RestBP'].mean()) / data['RestBP'].std()
data['Chol_ZScore'] = (data['Chol'] - data['Chol'].mean()) / data['Chol'].std()
data['MaxHR_ZScore'] = (data['MaxHR'] - data['MaxHR'].mean()) / data['MaxHR'].std()

# Decimal Scaling Normalization
data['RestBP_Decimal'] = data['RestBP'] / (10 ** len(str(data['RestBP'].max())))
data['Chol_Decimal'] = data['Chol'] / (10 ** len(str(data['Chol'].max())))
data['MaxHR_Decimal'] = data['MaxHR'] / (10 ** len(str(data['MaxHR'].max())))

# Display the normalized attributes
print("\nNormalized Attributes:")
print(data[['RestBP_MinMax', 'Chol_MinMax', 'MaxHR_MinMax',
            'RestBP_ZScore', 'Chol_ZScore', 'MaxHR_ZScore',
            'RestBP_Decimal', 'Chol_Decimal', 'MaxHR_Decimal']].head())
