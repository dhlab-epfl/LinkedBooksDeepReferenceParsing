import random

import numpy as np
import time

# Python objects
import pickle


# Plot
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# CRF
import sklearn_crfsuite
from sklearn_crfsuite 	import scorers, metrics
from sklearn.metrics 	import make_scorer, confusion_matrix
from sklearn.externals 	import joblib
from sklearn.model_selection import RandomizedSearchCV

# For model validation
import scipy


# Utils functions
from code.feature_extraction_supporting_functions_words import *
from code.feature_extraction_words import *
from code.utils import *



# Load validation data
window = 2
X_valid_w, valid_t1, valid_t2, valid_t3 = load_data("../dataset/clean_valid.txt")
X_valid  = [[word2features(text, i, window=window) for i in range(len(text))] for text in X_valid_w]



# TASK 1
y_valid = valid_t1
crf = pickle.load(open("models/crf_t1.pkl", "rb" ))
print(crf)
y_pred = crf.predict(X_valid)
print(metrics.flat_classification_report(
    y_valid, y_pred, digits=6
))

# Task 2
y_valid = valid_t2
crf = pickle.load(open("models/crf_t2.pkl", "rb" ))
print(crf)
y_pred = crf.predict(X_valid)
print(metrics.flat_classification_report(
    y_valid, y_pred, digits=6
))


# Task 3
y_valid = valid_t3
crf = pickle.load(open("models/crf_t3.pkl", "rb" ))
print(crf)
y_pred = crf.predict(X_valid)
print(metrics.flat_classification_report(
    y_valid, y_pred, digits=6
))
