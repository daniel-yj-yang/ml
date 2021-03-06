#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 15:32:23 2020

@author: daniel
"""

from machlearn import model_evaluation as me

from itertools import cycle
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
#from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
import nltk
import numpy as np
import matplotlib.pyplot as plt

#plt.rcParams.update({'font.size': 20})
#plt.rcParams.update({'figure.figsize': (10, 8)})

######################################################################################################

# 1. Multinomial Naive Bayes classifier

# https://stackabuse.com/the-naive-bayes-algorithm-in-python-with-scikit-learn/

########################################################################################
# Loading the Data

import pandas as pd

# Data Source: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
df = pd.read_table('/Users/daniel/Data-Science/DAta/Spam/SMS-Spam-Collection/SMSSpamCollection',
                   sep='\t',
                   header=None,
                   names=['label', 'message'])

########################################################################################
# Pre-processing

# First, we have to convert the labels from strings to binary values for our classifier:
df['label'] = df.label.map({'ham': 0, 'spam': 1})

# Second, convert all characters in the message to lower case:
df['message'] = df.message.map(lambda x: x.lower())

# Third, remove any punctuation:
df['message'] = df.message.str.replace('[^\w\s]', '')

# Fourth, tokenize the messages into into single words using nltk.
# nltk.download()

df['message'] = df['message'].apply(nltk.word_tokenize)

# Fifth, we will perform some word stemming.
# The idea of stemming is to normalize our text for all variations of words carry the same meaning, regardless of the tense.

stemmer = PorterStemmer()

df['message'] = df['message'].apply(lambda x: [stemmer.stem(y) for y in x])

# Finally, we will transform the data into occurrences, which will be the features that we will feed into our model:

# This converts the list of words into space-separated strings
df['message'] = df['message'].apply(lambda x: ' '.join(x))

count_vect = CountVectorizer()
counts = count_vect.fit_transform(df['message'])

# We could leave it as the simple word-count per message, but it is better to use Term Frequency Inverse Document Frequency, more known as tf-idf:

transformer = TfidfTransformer().fit(counts)

counts = transformer.transform(counts)

########################################################################################
# Training the Model


X_train, X_test, y_train, y_test = train_test_split(
    counts, df['label'], test_size=0.25, random_state=123)


model = MultinomialNB().fit(X_train, y_train)

########################################################################################
# Evaluating the Model

y_score = model.predict_proba(X_test)
y_pred = model.predict(X_test)
#print(np.mean(y_pred == y_test))

# comparing actual response values (y_test) with predicted response values (y_pred)
print("Multinomial Naive Bayes model accuracy(in %): {:0.2%}".format(
    metrics.accuracy_score(y_test, y_pred)))

# some references:
# https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
# https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
me.plot_confusion_matrix(y_test, y_pred, y_classes=('ham (y=0)', 'spam (y=1)'))

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
me.plot_ROC_and_PR_curves(fitted_model=model, X=X_test,
                          y_true=y_test, y_pred_score=y_score[:, 1], y_pos_label=1, model_name='Multinomial NB')


# more examples: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html


######################################################################################################
######################################################################################################
######################################################################################################

# 2. Gaussian Naive Bayes classifier

# https://www.geeksforgeeks.org/naive-bayes-classifiers/

# load the iris dataset
iris = load_iris()

# store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=123)

# training the model on training set
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# making predictions on the testing set
y_score = gnb.predict_proba(X_test)
y_pred = gnb.predict(X_test)
print(np.mean(y_pred == y_test))

# comparing actual response values (y_test) with predicted response values (y_pred)
print("Gaussian Naive Bayes model accuracy(in %): {:0.2%}".format(
    metrics.accuracy_score(y_test, y_pred)))

# https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
me.plot_confusion_matrix(y_test, y_pred, y_classes=np.unique(iris.target_names))

######################################################################################################
# Stop here

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# But this example is to binarize the y first...


n_classes = 3

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_test, y_score[:, i], pos_label=2)
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(figsize=(12, 10))

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
