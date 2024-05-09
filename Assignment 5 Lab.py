# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 19:10:38 2024

@author: Gaurav Bombale
"""

'''
5. Consider a suitable text dataset. Remove stop words, apply stemming and feature selection 
techniques to represent documents as vectors. Classify documents and evaluate precision, recall. 
(For Ex: Movie Review Dataset)
'''
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Load the Kaggle IMDB movie review dataset
data = pd.read_csv("IMDB Dataset.csv")

# Display the first few rows of the dataset to understand its structure
print("Dataset structure:")
print(data.head())

# Preprocessing: Remove stop words and apply stemming
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    words = word_tokenize(text.lower())  # Tokenize the text into words and convert to lowercase
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words]  # Stemming and remove stop words
    return " ".join(filtered_words)  # Join the words back into a string

# Apply the preprocessing function to each review in the dataset
data['processed_review'] = data['review'].apply(preprocess_text)

# Feature representation using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limiting the number of features to 5000
X = vectorizer.fit_transform(data['processed_review'])  # Transform the preprocessed text data into TF-IDF vectors

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['sentiment'], test_size=0.2, random_state=42)

# Classification using Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

# Evaluation: Precision and Recall
precision = precision_score(y_test, y_pred, pos_label='positive')
recall = recall_score(y_test, y_pred, pos_label='positive')

# Display the precision and recall scores
print("Precision:", precision)
print("Recall:", recall)


'''
A5
'''
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, chi2
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.datasets import load_files

# Load the IMDb movie review dataset
movie_reviews_data = load_files('path_to_dataset', shuffle=True)

# Extract features (reviews) and target labels (positive/negative sentiment)
X, y = movie_reviews_data.data, movie_reviews_data.target

# Preprocessing: Remove stop words and apply stemming
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

X_processed = [preprocess_text(text.decode('utf-8')) for text in X]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Define a pipeline for text classification
pipeline = Pipeline([
    ('vect', CountVectorizer()),  # Convert text to word count vectors
    ('tfidf', TfidfTransformer()),  # Convert word counts to TF-IDF scores
    ('feat_select', SelectKBest(chi2, k=1000)),  # Select top k features using chi-squared test
    ('clf', LogisticRegression()),  # Classifier (Logistic Regression)
])

# Train the classifier
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)







