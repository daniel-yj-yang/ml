#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 23:20:59 2020

@author: daniel
"""


# https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py

from machlearn import model_evaluation as me
from textblob import TextBlob
from sklearn.model_selection import train_test_split, GridSearchCV # StratifiedKFold, cross_val_score,
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix # , f1_score, accuracy_score
from sklearn.svm import SVC #, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#import sklearn
import csv
import pickle
import getopt
import sys
import os
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

#plt.rcParams.update({'font.size': 20})
#plt.rcParams.update({'figure.figsize': (8, 6)})


##############################################################################################################################

# modified from https://www.panggi.com/articles/sms-spam-filter-using-scikit-learn-and-textblob/
# converted from python2 to python3


# Imports

#import nltk

# Dataset
MESSAGES = pd.read_csv('/Users/daniel/Data-Science/Data/Spam/SMS-Spam-Collection/SMSSpamCollection',
                       sep='\t', quoting=csv.QUOTE_NONE, names=["label", "message"])

# Preprocessing


def tokens(message):
    message = str(message)  # , 'utf8')
    return TextBlob(message).words


def lemmas(message):
    message = str(message).lower()  # ), 'utf8').lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]

# Training


def train_multinomial_nb(messages):
    # split dataset for cross validation
    msg_train, msg_test, label_train, label_test = train_test_split(
        messages['message'], messages['label'], test_size=0.2)
    # create pipeline
    pipeline = Pipeline([('bow', CountVectorizer(analyzer=lemmas)),
                         ('tfidf', TfidfTransformer()), ('classifier', MultinomialNB())])
    # pipeline parameters to automatically explore and tune
    params = {
        'tfidf__use_idf': (True, False),
        'bow__analyzer': (lemmas, tokens),
    }
    grid = GridSearchCV(
        pipeline,
        params,  # parameters to tune via cross validation
        refit=True,  # fit using all data, on the best detected classifier
        n_jobs=-1,
        scoring='accuracy',
        cv=5,  # StratifiedKFold(n_splits=5).split(label_train),
    )
    # train
    nb_detector = grid.fit(msg_train, label_train)
    print("")
    y_test = label_test
    y_pred = nb_detector.predict(msg_test)
    y_score = nb_detector.predict_proba(msg_test)

    print(":: Confusion Matrix")
    print("")
    cf_matrix = confusion_matrix(y_test, y_pred)
    print(cf_matrix)

    # https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
    categories = ['Ham (y=0)', 'Spam (y=1)']
    me.plot_confusion_matrix(cm = cf_matrix,
                             y_classes = categories)

    # comparing actual response values (y_test) with predicted response values (y_pred)
    from sklearn import metrics
    print("Model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

    fpr, tpr, thresholds = metrics.roc_curve(
        y_test, y_score[:, 1], pos_label='spam')
    auc = metrics.roc_auc_score(y_test, y_score[:, 1])
    print(auc)

    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    plt.figure(figsize=(8, 6))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw,
             label="Multinomial NB (AUC = %0.2f)" % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

    print("")
    print(":: Classification Report")
    print("")
    print(classification_report(y_test, y_pred))
    # save model to pickle file
    file_name = '/Users/daniel/Data-Science/Data/Spam/SMS-Spam-Collection/ml_models/sms_spam_nb_model.pkl'
    with open(file_name, 'wb') as fout:
        pickle.dump(nb_detector, fout)
    print('model written to: ' + file_name)


def train_svm(messages):
    # split dataset for cross validation
    msg_train, msg_test, label_train, label_test = train_test_split(
        messages['message'], messages['label'], test_size=0.2)
    # create pipeline
    pipeline = Pipeline([('bow', CountVectorizer(analyzer=lemmas)),
                         ('tfidf', TfidfTransformer()), ('classifier', SVC(probability=True))])
    # pipeline parameters to automatically explore and tune
    params = [
        {'classifier__C': [0.1, 1, 10, 100, 1000],
            'classifier__kernel': ['linear']},
        {'classifier__C': [0.1, 1, 10, 100, 1000], 'classifier__gamma': [
            0.001, 0.0001], 'classifier__kernel': ['rbf']},
    ]
    grid = GridSearchCV(
        pipeline,
        param_grid=params,  # parameters to tune via cross validation
        refit=True,  # fit using all data, on the best detected classifier
        n_jobs=-1,
        scoring='accuracy',
        cv=5  # StratifiedKFold(label_train, n_splits=5),
    )
    # train
    svm_detector = grid.fit(msg_train, label_train)
    print("")
    y_test = label_test
    y_pred = svm_detector.predict(msg_test)
    y_score = svm_detector.predict_proba(msg_test)

    print(":: Confusion Matrix")
    print("")
    cf_matrix = confusion_matrix(y_test, y_pred)
    print(cf_matrix)

    # https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
    categories = ['Ham (y=0)', 'Spam (y=1)']
    me.plot_confusion_matrix(cm = cf_matrix,
                          y_classes = categories)

    # comparing actual response values (y_test) with predicted response values (y_pred)
    from sklearn import metrics
    print("Model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

    fpr, tpr, thresholds = metrics.roc_curve(
        y_test, y_score[:, 1], pos_label='spam')
    auc = metrics.roc_auc_score(y_test, y_score[:, 1])
    print(auc)

    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    plt.figure(figsize=(8, 6))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw,
             label="SVM (AUC = %0.2f)" % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

    print("")
    print(":: Classification Report")
    print("")
    print(classification_report(y_test, y_pred))
    # save model to pickle file
    file_name = '/Users/daniel/Data-Science/Data/Spam/SMS-Spam-Collection/ml_models/sms_spam_svm_model.pkl'
    with open(file_name, 'wb') as fout:
        pickle.dump(svm_detector, fout)
    print('model written to: ' + file_name)


def main(argv):
  # check if models exist, if not run training
    if(os.path.isfile('/Users/daniel/Data-Science/Data/Spam/SMS-Spam-Collection/ml_models/sms_spam_nb_model.pkl') == False):
        print("")
        print("Creating Naive Bayes Model.....")
        train_multinomial_nb(MESSAGES)

    if(os.path.isfile('/Users/daniel/Data-Science/Data/Spam/SMS-Spam-Collection/ml_models/sms_spam_svm_model.pkl') == False):
        print("")
        print("Creating SVM Model.....")
        train_svm(MESSAGES)

    #inputmessage = ''
    try:
        opts, args = getopt.getopt(argv, "hm:", ["message="])
    except getopt.GetoptError:
        print('SMS_spam_classifier.py -m <message string>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('SMS_spam_classifier.py -m <message string>')
            sys.exit()
        elif opt in ("-m", "--message"):
            prediction = predict(arg)
            print('This message is predicted by', prediction)
        else:
            print('SMS_spam_classifier.py -m <message string>')
            sys.exit()


def predict(message):
    nb_detector = pickle.load(open(
        '/Users/daniel/Data-Science/Data/Spam/SMS-Spam-Collection/ml_models/sms_spam_nb_model.pkl', 'rb'))
    svm_detector = pickle.load(open(
        '/Users/daniel/Data-Science/Data/Spam/SMS-Spam-Collection/ml_models/sms_spam_svm_model.pkl', 'rb'))

    nb_predict = nb_detector.predict([message])[0]
    svm_predict = svm_detector.predict([message])[0]

    return 'SVM as ' + svm_predict + ' and Naive Bayes as ' + nb_predict


if __name__ == "__main__":
    main(sys.argv[1:])
