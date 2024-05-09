# -*- coding: utf-8 -*-
"""
Created on Tue Feb 29 08:54:22 2024

@author: Gaurav Bombale
"""
'''
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
url = "https://raw.githubusercontent.com/zhaoyingzhu/heart-disease-prediction/master/dataset.csv"
heart_data = pd.read_csv(url)

# Display the first few rows of the dataset
print(heart_data.head())

# a) Find standard deviation, variance of every numerical attribute
std_dev = heart_data.std()
variance = heart_data.var()
print("\nStandard Deviation of each attribute:")
print(std_dev)
print("\nVariance of each attribute:")
print(variance)

# b) Find covariance and perform Correlation analysis using Correlation coefficient
covariance_matrix = heart_data.cov()
correlation_matrix = heart_data.corr(method='pearson')
print("\nCovariance Matrix:")
print(covariance_matrix)
print("\nCorrelation Matrix:")
print(correlation_matrix)

# c) How many independent features are present in the given dataset?
# We can use correlation matrix to identify independent features.
# A feature is independent if its correlation coefficient with all other features is close to 0.
independent_features = correlation_matrix[(correlation_matrix.abs() < 0.1) & (correlation_matrix.abs() != 1)].dropna(how='all', axis=0).dropna(how='all', axis=1)
print("\nIndependent features:")
print(independent_features)

# d) Can we identify unwanted features?
# Unwanted features are those with high correlation with other features or those with low correlation with the target variable.
# We can use correlation matrix to identify such features.
# For simplicity, let's consider features with correlation coefficient above 0.5 as potentially unwanted.
unwanted_features = correlation_matrix[(correlation_matrix.abs() > 0.5) & (correlation_matrix.abs() != 1)].dropna(how='all', axis=0).dropna(how='all', axis=1)
print("\nUnwanted features:")
print(unwanted_features)

# e) Perform the data discretization using equi frequency binning method on age attribute
bins = pd.qcut(heart_data['age'], q=5, labels=False)
heart_data['age_discretized'] = bins
print("\nData after discretization:")
print(heart_data[['age', 'age_discretized']].head())

# f) Normalize RestBP, chol, and MaxHR attributes using min-max normalization, Z-score normalization, and decimal scaling normalization
def min_max_normalization(x):
    return (x - x.min()) / (x.max() - x.min())

def z_score_normalization(x):
    return (x - x.mean()) / x.std()

def decimal_scaling_normalization(x):
    max_abs_value = x.abs().max()
    return x / (10 ** np.ceil(np.log10(max_abs_value)))

# Select attributes for normalization
attributes_to_normalize = ['RestBP', 'chol', 'MaxHR']

# Min-max normalization
heart_data_min_max_normalized = heart_data.copy()
heart_data_min_max_normalized[attributes_to_normalize] = heart_data_min_max_normalized[attributes_to_normalize].apply(min_max_normalization)

# Z-score normalization
heart_data_z_score_normalized = heart_data.copy()
heart_data_z_score_normalized[attributes_to_normalize] = heart_data_z_score_normalized[attributes_to_normalize].apply(z_score_normalization)

# Decimal scaling normalization
heart_data_decimal_scaling_normalized = heart_data.copy()
heart_data_decimal_scaling_normalized[attributes_to_normalize] = heart_data_decimal_scaling_normalized[attributes_to_normalize].apply(decimal_scaling_normalization)

# Display the normalized data
print("\nData after normalization (Min-max normalization):")
print(heart_data_min_max_normalized[attributes_to_normalize].head())
print("\nData after normalization (Z-score normalization):")
print(heart_data_z_score_normalized[attributes_to_normalize].head())
print("\nData after normalization (Decimal scaling normalization):")
print(heart_data_decimal_scaling_normalized[attributes_to_normalize].head())

