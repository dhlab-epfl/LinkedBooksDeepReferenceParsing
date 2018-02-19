import random
import numpy as np
import tensorflow

# Seed
random.seed(42)
np.random.seed(42)
tensorflow.set_random_seed(42)

# Models and Utils scripts
from code.models import *
from code.utils import *


# Load entire data
X_train_w, y_train1_w, y_train2_w, y_train3_w 	= load_data("dataset/clean_train.txt")	# Training data
X_test_w,  y_test1_w,  y_test2_w,  y_test3_w 	= load_data("dataset/clean_test.txt")	# Testing data
X_valid_w, y_valid1_w, y_valid2_w, y_valid3_w 	= load_data("dataset/clean_valid.txt")	# Validation data


# Merge digits under the same word
digits_word = "$NUM$" 
X_train_w, X_test_w, X_valid_w = mergeDigits([X_train_w, X_test_w, X_valid_w], digits_word)

# Compute indexes for words+labels in the training data
ukn_words = "out-of-vocabulary"   # Out-of-vocabulary words entry in the "words to index" dictionary
word2ind,   ind2word   =  indexData_x(X_train_w, ukn_words)
label2ind1, ind2label1 =  indexData_y(y_train1_w)
label2ind2, ind2label2 =  indexData_y(y_train2_w)
label2ind3, ind2label3 =  indexData_y(y_train3_w)

print(ind2label1)
print(ind2label2)
print(ind2label3)



# Convert data into indexes data
maxlen  = max([len(xx) for xx in X_train_w])
padding_style   = 'pre'                 # 'pre' or 'post': Style of the padding, in order to have sequence of the same size
X_train   = encodePadData_x(X_train_w,  word2ind,   maxlen, ukn_words, padding_style)
X_test    = encodePadData_x(X_test_w,   word2ind,   maxlen, ukn_words, padding_style)
X_valid   = encodePadData_x(X_valid_w,  word2ind,   maxlen, ukn_words, padding_style)

y_train1  = encodePadData_y(y_train1_w, label2ind1, maxlen, padding_style)
y_test1   = encodePadData_y(y_test1_w,  label2ind1, maxlen, padding_style)
y_valid1  = encodePadData_y(y_valid1_w, label2ind1, maxlen, padding_style)

y_train2  = encodePadData_y(y_train2_w, label2ind2, maxlen, padding_style)
y_test2   = encodePadData_y(y_test2_w,  label2ind2, maxlen, padding_style)
y_valid2  = encodePadData_y(y_valid2_w, label2ind2, maxlen, padding_style)

y_train3  = encodePadData_y(y_train3_w, label2ind3, maxlen, padding_style)
y_test3   = encodePadData_y(y_test3_w,  label2ind3, maxlen, padding_style)
y_valid3  = encodePadData_y(y_valid3_w, label2ind3, maxlen, padding_style)



# Create the character level data
char2ind, maxWords, maxChar = characterLevelIndex(X_train_w, digits_word)
X_train_char = characterLevelData(X_train_w, char2ind, maxWords, maxChar, digits_word, padding_style)
X_test_char  = characterLevelData(X_test_w,  char2ind, maxWords, maxChar, digits_word, padding_style)
X_valid_char = characterLevelData(X_valid_w, char2ind, maxWords, maxChar, digits_word, padding_style)


# Training, Tesing and Validation data for the model (word emb + char features)
X_training = [X_train, X_train_char]
X_testing = [X_test, X_test_char]
X_validation = [X_valid, X_valid_char]


# Model parameters
epoch = 25
batch = 100
dropout = 0.5
lstm_size = 200



model_name = "task1"
BiLSTM_model(model_name, True, "crf",
	  X_training, X_testing, word2ind, maxWords,
	  [y_train1], [y_test1], [ind2label1],
	  validation=True, X_valid=X_validation, y_valid=[y_valid1],
	  pretrained_embedding=True, word_embedding_size=300,
	  maxChar=maxChar, char_embedding_type="BILSTM", char2ind=char2ind, char_embedding_size=100,
	  lstm_hidden=lstm_size, nbr_epochs=epoch, batch_size=batch, dropout=dropout,
	  gen_confusion_matrix=True, early_stopping_patience=5
	)

print("=====")

model_name = "task2"
BiLSTM_model(model_name, True, "crf",
	  X_training, X_testing, word2ind, maxWords,
	  [y_train2], [y_test2], [ind2label2],
	  validation=True, X_valid=X_validation, y_valid=[y_valid2],
	  pretrained_embedding=True, word_embedding_size=300,
	  maxChar=maxChar, char_embedding_type="BILSTM", char2ind=char2ind, char_embedding_size=100,
	  lstm_hidden=lstm_size, nbr_epochs=epoch, batch_size=batch, dropout=dropout,
	  gen_confusion_matrix=True, early_stopping_patience=5
	)

print("=====")

model_name = "task3"
BiLSTM_model(model_name, True, "crf",
	  X_training, X_testing, word2ind, maxWords,
	  [y_train3], [y_test3], [ind2label3],
	  validation=True, X_valid=X_validation, y_valid=[y_valid3],
	  pretrained_embedding=True, word_embedding_size=300,
	  maxChar=maxChar, char_embedding_type="BILSTM", char2ind=char2ind, char_embedding_size=100,
	  lstm_hidden=lstm_size, nbr_epochs=epoch, batch_size=batch, dropout=dropout,
	  gen_confusion_matrix=True, early_stopping_patience=5
	)


print("Done.")
