#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 15:32:23 2020

@author: daniel
"""

from machlearn import model_evaluation as me
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
import nltk
#nltk.download()

df['message'] = df['message'].apply(nltk.word_tokenize)

# Fifth, we will perform some word stemming.
# The idea of stemming is to normalize our text for all variations of words carry the same meaning, regardless of the tense.
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

df['message'] = df['message'].apply(lambda x: [stemmer.stem(y) for y in x])

# Finally, we will transform the data into occurrences, which will be the features that we will feed into our model:
from sklearn.feature_extraction.text import CountVectorizer

# This converts the list of words into space-separated strings
df['message'] = df['message'].apply(lambda x: ' '.join(x))

count_vect = CountVectorizer()
counts = count_vect.fit_transform(df['message'])

# We could leave it as the simple word-count per message, but it is better to use Term Frequency Inverse Document Frequency, more known as tf-idf:
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer().fit(counts)

counts = transformer.transform(counts)

########################################################################################
# Training the Model

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(counts, df['label'], test_size=0.25, random_state=123)

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB().fit(X_train, y_train)

########################################################################################
# Evaluating the Model

y_score = model.predict_proba(X_test)
y_pred = model.predict(X_test)
#print(np.mean(y_pred == y_test))

# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("Multinomial Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(y_test, y_pred, target_names=('ham','spam')))
cf_matrix = confusion_matrix(y_test, y_pred)


# some references
# https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
# https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
me.plot_confusion_matrix(cf_matrix,
                      y_classes=['Ham (y=0)', 'Spam (y=1)'],
                      figsize=(12,10))

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score[:,1], pos_label=1)
auc = metrics.roc_auc_score(y_test, y_score[:,1])
#auc = np.trapz(tpr,fpr) # alternatively

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
fig = plt.figure(figsize=(10,8))
lw = 2
plt.plot(fpr,tpr,color = 'darkorange', lw=lw, label="Multinomial NB (AUC = %0.2f)" % auc )
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate = p(y_pred=1 | y_actual=0)')
plt.ylabel('True Positive Rate = p(y_pred=1 | y_actual=1)')
plt.title('ROC Curve')
fig.tight_layout()
plt.show()

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_score[:,1])

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

from sklearn.metrics import plot_precision_recall_curve

disp = plot_precision_recall_curve(model, X_test, y_test)
#disp.figsize = (10,10)
disp.ax_.set_title('Precision-Recall Curve')
disp.ax_.set_xlabel('Recall p(y_pred=1 | y_actual=1)')
disp.ax_.set_ylabel('Precision p(y_actual=1 | y_pred=1)')
#disp.plot()



######################################################################################################
######################################################################################################
######################################################################################################

# 2. Gaussian Naive Bayes classifier

# https://www.geeksforgeeks.org/naive-bayes-classifiers/

# load the iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

# store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target

# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

# training the model on training set
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# making predictions on the testing set
y_score = gnb.predict_proba(X_test)
y_pred = gnb.predict(X_test)
print(np.mean(y_pred == y_test))

# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

from sklearn.metrics import confusion_matrix
cf_matrix_3x3 = confusion_matrix(y_test, y_pred)

# https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
me.plot_confusion_matrix(cf_matrix_3x3,
                         y_classes = iris.target_names,
                         figsize=(9,7))

######################################################################################################
# Stop here

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# But this example is to binarize the y first...

from itertools import cycle

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
plt.figure(figsize=(12,10))

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
