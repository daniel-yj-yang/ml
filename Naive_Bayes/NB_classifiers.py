#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 15:32:23 2020

@author: daniel
"""

######################################################################################################

# Multinomial Naive Bayes classifier

# https://stackabuse.com/the-naive-bayes-algorithm-in-python-with-scikit-learn/

import pandas as pd

# Data Source: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
df = pd.read_table('/Users/daniel/Data-Science/DAta/Spam/SMS-Spam-Collection/SMSSpamCollection',
                   sep='\t',
                   header=None,
                   names=['label', 'message'])

df['label'] = df.label.map({'ham': 0, 'spam': 1})

df['message'] = df.message.map(lambda x: x.lower())

df['message'] = df.message.str.replace('[^\w\s]', '')

import nltk
#nltk.download()

df['message'] = df['message'].apply(nltk.word_tokenize)

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

df['message'] = df['message'].apply(lambda x: [stemmer.stem(y) for y in x])

from sklearn.feature_extraction.text import CountVectorizer

# This converts the list of words into space-separated strings
df['message'] = df['message'].apply(lambda x: ' '.join(x))

count_vect = CountVectorizer()
counts = count_vect.fit_transform(df['message'])

from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer().fit(counts)

counts = transformer.transform(counts)

# Training the Model

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(counts, df['label'], test_size=0.1, random_state=73)

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB().fit(X_train, y_train)

# Evaluating the Model

import numpy as np

predicted = model.predict(X_test)

print(np.mean(predicted == y_test))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, predicted))

######################################################################################################

# Gaussian Naive Bayes classifier

# https://www.geeksforgeeks.org/naive-bayes-classifiers/

# load the iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

# store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target

# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# training the model on training set
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# making predictions on the testing set
y_pred = gnb.predict(X_test)

# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)