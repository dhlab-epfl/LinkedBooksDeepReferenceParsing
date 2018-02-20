import random

import numpy as np
import time

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


# Load entire data
X_train_w, train_t1, train_t2, train_t3 = load_data("../dataset/clean_train.txt")
X_test_w, test_t1, test_t2, test_t3= load_data("../dataset/clean_test.txt")


for task in ["t3", "t2", "t1"]: #Ordered according to increase computation time

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



	# CRF Model : Fine-tuning c1 and c2

	# Parameters search (Based on https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#hyperparameter-optimization)
	crf = sklearn_crfsuite.CRF( 
		max_iterations=100,
		algorithm = 'lbfgs',
		all_possible_transitions=False
	)

	params_space = {
		'c1': scipy.stats.expon(scale=0.5),
		'c2': scipy.stats.expon(scale=0.05)
	}

	scorer = make_scorer(metrics.flat_f1_score, average='weighted')
		
	# search
	rs = RandomizedSearchCV(crf, params_space, 
							cv=3, 
							verbose=1, 
							n_jobs=-10, 
							n_iter=5, 
							scoring=scorer)
	rs.fit(X_train, y_train)

	print('best params:', rs.best_params_)
	print('best CV score:', rs.best_score_)


	# Create score plot
	_x = [s.parameters['c1'] for s in rs.grid_scores_]
	_y = [s.parameters['c2'] for s in rs.grid_scores_]
	_c = [s.mean_validation_score for s in rs.grid_scores_]

	fig = plt.figure()
	fig.set_size_inches(12, 12)
	ax = plt.gca()
	ax.set_xlabel('C1')
	ax.set_ylabel('C2')
	ax.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(min(_c), max(_c)))
	ax.scatter(_x, _y, c=_c, s=60, cmap="bwr_r")
	print("F1 scores: Dark blue => {:0.4}, dark red => {:0.4}".format(min(_c), max(_c)))
	plt.savefig("plots/plot_fine_tuning_task_{0}".format(task))
	#Save plot




	# Testing with best parameters
	y_pred = rs.best_estimator_.predict(X_test)
	print('best params:', rs.best_params_)
	print('best CV score:', rs.best_score_)
	print(metrics.flat_classification_report(
	    y_test, y_pred, digits=3
	))


	# Save best model
	joblib.dump(rs.best_estimator_,'models/crf_{0}.pkl'.format(task))


	# Close file
	closePrintToFile(file, stdout_original)
	print("=========================== Task {0} ========================= End:{1}".format(task,  time.strftime("%D %H:%M:%S")))
