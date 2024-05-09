'''
 Classify iris plants into three species use following dataset 
https://www.kaggle.com/datasets/uciml/iris (Give comparative analysis of any three classification 
techniques based on accuracy).
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
iris_data = pd.read_csv("iris.csv")

# EDA
# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(iris_data.head())

# Summary statistics
print("\nSummary statistics of the dataset:")
print(iris_data.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(iris_data.isnull().sum())

# Check class distribution
print("\nClass distribution:")
print(iris_data['species'].value_counts())

# Pairplot for visualization
sns.pairplot(iris_data, hue='species')
plt.show()

# Boxplot for visualization
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris_data, x='species', y='petal_length')
plt.title('Boxplot of Petal Length by Species')
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = iris_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Classification
# Split the dataset into features and target variable
X = iris_data.drop('species', axis=1)
y = iris_data['species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train)
log_reg_pred = log_reg_model.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Comparative analysis
print("\nLogistic Regression Accuracy:", log_reg_accuracy)
print("Decision Tree Accuracy:", dt_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
