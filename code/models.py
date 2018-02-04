# -*- coding: utf-8 -*-

"""
Functions for building Keras models
"""

import os
import random
import numpy as np
import tensorflow
random.seed(42)
np.random.seed(42)
tensorflow.set_random_seed(42)

# Keras function
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, Input, TimeDistributed, Flatten, Convolution1D, MaxPooling1D, concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils

from sklearn.metrics import confusion_matrix
from sklearn_crfsuite import metrics

# Utils script
from code.utils import *



def BiLSTM_model(filename, train, output,
              X_train, X_test, word2ind, maxWords,
              y_train, y_test, ind2label,
              validation=False, X_valid=None, y_valid=None,
              word_embeddings=True, pretrained_embedding="", word_embedding_size=100,
              maxChar=0, char_embedding_type="", char2ind="", char_embedding_size=50,
              lstm_hidden=32, nbr_epochs=5, batch_size=32, dropout=0, optimizer='rmsprop', early_stopping_patience=-1,
              folder_path="model_results", gen_confusion_matrix=False
            ):    
    """
        Build, train and test a BiLSTM Keras model. Works for multi-tasking learning.
        The model architecture looks like:
            
            - Words representations: 
                - Word embeddings
                - Character-level representation [Optional]
            - Dropout
            - Bidirectional LSTM
            - Dropout
            - Softmax/CRF for predictions


        :param filename: File to redirect the printing
        :param train: Boolean if the model must be trained or not. If False, the model's wieght are expected to be stored in "folder_path/filename/filename.h5" 
        :param otput: "crf" or "softmax". Type of prediction layer to use
        
        :param X_train: Data to train the model
        :param X_test: Data to test the model
        :param word2ind: Dictionary containing all words in the training data and a unique integer per word
        :param maxWords: Maximum number of words in a sequence 

        :param y_train: Labels to train the model for the prediction task
        :param y_test: Labels to test the model for the prediction task
        :param ind2label: Dictionary where all labels for task 1 are mapped into a unique integer

        :param validation: Boolean. If true, the validation score will be computed from 'X_valid' and 'y_valid'
        :param X_valid: Optional. Validation dataset
        :param y_valid: Optional. Validation dataset labels

        :param word_embeddings: Boolean value. Add word embeddings into the model.
        :param pretrained_embedding: Use the pretrained word embeddings. 
                                     Three values: 
                                            - "":    Do not use pre-trained word embeddings (Default)
                                            - False: Use the pre-trained embedding vectors as the weights in the Embedding layer
                                            - True:  Use the pre-trained embedding vectors as weight initialiers. The Embedding layer will still be trained.
        :param word_embedding_size: Size of the pre-trained word embedding to use (100 or 300)

        :param maxChar: The maximum numbers of characters in a word. If set to 0, the model will not use character-level representations of the words
        :param char_embedding_type: Type of model to use in order to compute the character-level representation of words: Two values: "CNN" or "BILSTM"
        :param char2ind: A dictionary where each character is maped into a unique integer
        :param char_embedding_size: size of the character-level word representations

        :param lstm_hidden: Dimentionality of the LSTM output space
        :param nbr_epochs: Number of epochs to train the model
        :param batch_size: Size of batches while training the model
        :param dropout: Rate to apply for each Dropout layer in the model
        :param optimizer: Optimizer to use while compiling the model
        :param early_stopping_patience: Number of continuous tolerated epochs without improvement during training.

        :param folder_path: Path to the directory storing all to-be-generated files
        :param gen_confusion_matrix: Boolean value. Generated confusion matrices or not.


        :return: The classification scores for both tasks.
    """      
    print("====== {0} start ======".format(filename))
    end_string = "====== {0} end ======".format(filename)
    
    # Create directory to store results
    os.makedirs(folder_path+"/"+filename)
    filepath = folder_path+"/"+filename+"/"+filename

    # Set print outputs file
    file, stdout_original = setPrintToFile("{0}.txt".format(filepath))    

    # Model params
    nbr_words = len(word2ind)+1
    out_size = len(ind2label)+1
    best_results = ""
    
    embeddings_list = []
    inputs = []

    # Input - Word Embeddings
    if word_embeddings:
        word_input = Input((maxWords,))
        inputs.append(word_input)
        if pretrained_embedding=="":
            word_embedding = Embedding(nbr_words, word_embedding_size)(word_input)
        else:
            # Retrieve embeddings
            embedding_matrix = word2VecEmbeddings(word2ind, word_embedding_size)
            word_embedding = Embedding(nbr_words, word_embedding_size, weights=[embedding_matrix], trainable=pretrained_embedding, mask_zero=False)(word_input)
        embeddings_list.append(word_embedding)

    # Input - Characters Embeddings
    if maxChar!=0:
        character_input     = Input((maxWords,maxChar,))
        char_embedding      = character_embedding_layer(char_embedding_type, character_input, maxChar, len(char2ind)+1, char_embedding_size) 
        embeddings_list.append(char_embedding)
        inputs.append(character_input)

    # Model - Inner Layers - BiLSTM with Dropout
    embeddings = concatenate(embeddings_list) if len(embeddings_list)==2 else embeddings_list[0]
    model = Dropout(dropout)(embeddings)
    model = Bidirectional(LSTM(lstm_hidden, return_sequences=True, dropout=dropout))(model)
    model = Dropout(dropout)(model)
    
    
    if output == "crf":
        # Output - CRF
        crfs = [[CRF(out_size),out_size] for out_size in [len(x)+1 for x in ind2label]]
        outputs = [x[0](Dense(x[1])(model)) for x in crfs]
        model_loss = [x[0].loss_function for x in crfs]
        model_metrics = [x[0].viterbi_acc for x in crfs]

    if output == "softmax":
        outputs = [Dense(out_size, activation='softmax')(model) for out_size in [len(x)+1 for x in ind2label]]
        model_loss = ['categorical_crossentropy' for x in outputs]
        model_metrics = None

    # Model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=model_loss, metrics=model_metrics, optimizer=get_optimizer(optimizer))
    print(model.summary(line_length=150),"\n\n\n\n")


    # Training Callbacks:
    callbacks = []
    value_to_monitor = 'val_f1'
    best_model_weights_path = "{0}.h5".format(filepath)
    
    #    1) Classifition scores
    classification_scores = Classification_Scores([X_train, y_train], ind2label, best_model_weights_path)
    callbacks.append(classification_scores)
    
    #    2) EarlyStopping
    if early_stopping_patience != -1:
        early_stopping = EarlyStopping(monitor=value_to_monitor, patience=early_stopping_patience, mode='max')
        callbacks.append(early_stopping)


    # Train
    if train:
        # Train the model. Keras's method argument 'validation_data' is referred as 'testing data' in this code.
        hist = model.fit(X_train, y_train, validation_data=[X_test, y_test], epochs=nbr_epochs, batch_size=batch_size, callbacks=callbacks, verbose=2)
        
        print()
        print('-------------------------------------------')
        print("Best F1 score:", early_stopping.best, "  (epoch number {0})".format(1+np.argmax(hist.history[value_to_monitor])))
        
        # Save Training scores
        save_model_training_scores("{0}".format(filepath), hist, classification_scores)
        
        # Print best testing classification report
        best_epoch = np.argmax(hist.history[value_to_monitor])
        print(classification_scores.test_report[best_epoch])

        
        # Best epoch results
        best_results = model_best_scores(classification_scores, best_epoch)

    # Load weigths from best training epoch into model
    save_load_utils.load_all_weights(model, best_model_weights_path)

    # Create confusion matrices
    if gen_confusion_matrix:
        for i, y_target in enumerate(y_test):
            # Compute predictions, flatten
            predictions, target = compute_predictions(model, X_test, y_target, ind2label[i])
            # Generate confusion matrices
            save_confusion_matrix(target, predictions,  list(ind2label[i].values()), "{0}_task_{1}_confusion_matrix_test".format(filepath,str(i+1)))


    # Validation dataset
    if validation:
        print()
        print("Validation dataset")
        print("======================")
        # Compute classification report 
        for i, y_target in enumerate(y_valid):
            # Compute predictions, flatten
            predictions, target = compute_predictions(model, X_valid, y_target, ind2label[i], nbrTask=i)

            # Only for multi-task
            if len(y_train) > 1:
                print("For task "+str(i+1)+"\n")
                print("====================================================================================")

            print("")
            print("With padding into account")
            print(metrics.flat_classification_report([target], [predictions], digits=4))
            print("")
            print('----------------------------------------------')
            print("")
            print("Without the padding:")
            print(metrics.flat_classification_report([target], [predictions], digits=4, labels=list(ind2label[i].values())))

            # Generate confusion matrices
            save_confusion_matrix(target, predictions,  list(ind2label[i].values()), "{0}_task_{1}_confusion_matrix_validation".format(filepath,str(i+1)))


    # Close file
    closePrintToFile(file, stdout_original)
    print(end_string)

    return best_results




def character_embedding_layer(layer_type, character_input, maxChar, nbr_chars, char_embedding_size, 
                            cnn_kernel_size=2, cnn_filters=30, lstm_units=50, dropout=0.5):
    """
        Return layer for computing the character-level representations of words.
        
        There is two type of architectures:

            Architecture CNN:
                - Character Embeddings
                - Dropout
                - Flatten
                - Convolution
                - MaxPool

            Architecture BILSTM:
                - Character Embeddings
                - Dropout
                - Flatten
                - Bidirectional LSTM

        :param layer_type: Model architecture to use "CNN" or "BILSTM"
        :param character_input: Keras Input layer, size of the input
        :param maxChar: The maximum numbers of characters in a word. If set to 0, the model will not use character-level representations of the words
        :param nbr_chars: Numbers of unique characters present in the data
        :param char_embedding_size: size of the character-level word representations
        :param cnn_kernel_size: For the CNN architecture, size of the kernel in the Convolution layer
        :param cnn_filters: For the CNN architecture, number of filters in the Convolution layer
        :param lstm_units: For the BILSTM architecture, dimensionality of the output LSTM space (half of the Bidirectinal LSTM output space)
        :param dropout: Rate to apply for each Dropout layer in the model

        :return: Character-level representation layers
    """

    embed_char_out      = TimeDistributed(Embedding(nbr_chars, char_embedding_size), name='char_embedding')(character_input)
    dropout             = Dropout(dropout)(embed_char_out)
    dropout             = TimeDistributed(Flatten())(dropout)
    
    if layer_type == "CNN":
        conv1d_out      = TimeDistributed(Convolution1D(kernel_size=cnn_kernel_size, filters=cnn_filters, padding='same'))(dropout)
        char_emb        = TimeDistributed(MaxPooling1D(maxChar))(conv1d_out)
    
    if layer_type == "BILSTM":
        char_emb        = Bidirectional(LSTM(lstm_units,return_sequences=True))(dropout)
    
    return char_emb




def get_optimizer(type, learning_rate=0.001, decay=0.0):
    """
        Return the optimizer needeed to compile Keras models.

        :param type: Type of optimizer. Two types supported: 'ADAM' and 'RMSprop'
        :param learning_rate: float >= 0. Learning rate.
        :pram decay:float >= 0. Learning rate decay over each update

        :return: The optimizer to use directly into keras model compiling function.
    """

    if type == "adam":
        return Adam(lr=learning_rate, decay=decay)

    if type == "rmsprop":
        return RMSprop(lr=learning_rate, decay=decay)