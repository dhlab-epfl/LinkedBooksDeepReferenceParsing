import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time

# CRF
import sklearn_crfsuite
from sklearn_crfsuite 	import scorers, metrics
from sklearn.metrics 	import make_scorer, confusion_matrix
from sklearn.externals 	import joblib


# Utils functions
from code.feature_extraction_supporting_functions_words import *
from code.feature_extraction_words import *
from code.utils import *


# Load entire data
X_train_w, train_t1, train_t2, train_t3 = load_data("../dataset/clean_train.txt")
X_test_w, test_t1, test_t2, test_t3= load_data("../dataset/clean_test.txt")


for task in ["t1", "t2", "t3"]:

	print("=========================== Task {0} ========================= Start:{1}".format(task,  time.strftime("%D %H:%M:%S")))
	# Set file
	file, stdout_original = setPrintToFile("results/CRF_model_task_{0}.txt".format(task))

	# Task data
	y_train = eval("train_"+task)
	y_test = eval("test_"+task)

	# Build CRF data format
	window = 2 # the window of dependance for the CRFs
	X_train = [[word2features(text, i, window=window) for i in range(len(text))] for text in X_train_w]
	X_test  = [[word2features(text, i, window=window) for i in range(len(text))] for text in X_test_w]


	print("Training data  - number of lines:  ", len(X_train))
	print("Testing  data  - number of lines:  ", len(X_test))
	print('----')
	print("Training data  - number of tokens:  ", len([x for y in X_train for x in y]))
	print("Testing  data  - number of tokens:  ", len([x for y in X_test  for x in y]))
	print()
	print()



	# CRF Model

	crf = sklearn_crfsuite.CRF( 
	    algorithm='lbfgs',
	    c1=0.1,
	    c2=0.1,
	    max_iterations=100,
	    all_possible_transitions=False
	)
	crf.fit(X_train, y_train)

	# Save CRF model
	joblib.dump(crf,'models/crf_{0}.pkl'.format(task))



	# Testing
	y_pred = crf.predict(X_test)
	print(metrics.flat_classification_report(
	    y_test, y_pred, digits=3
	))


	# Close file
	closePrintToFile(file, stdout_original)
	print("=========================== Task {0} ========================= End:{1}".format(task,  time.strftime("%D %H:%M:%S")))