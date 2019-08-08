# -*- coding: utf-8 -*-

"""
Support functions for dealing with data and building models
"""

import random
import numpy as np
import tensorflow
random.seed(42)
np.random.seed(42)
tensorflow.set_random_seed(42)

import sys
import csv
import itertools

from keras.callbacks import Callback
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.utils import save_load_utils
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn_crfsuite import metrics

# Plot
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt



def load_data(filepath):
    """
        Load and return the data stored in the given path.
        The data is structured as follows: 
            Each line contains four columns separated by a single space. 
            Each word has been put on a separate line and there is an empty line after each sentence. 
            The first item on each line is a word, the second, third and fourth are tags related to the word.
        Example:
            The sentence "L. Antonielli, Iprefetti dell' Italia napoleonica, Bologna 1983." is represented in the dataset as:
                L author b-secondary b-r
                . author i-secondary i-r
                Antonielli author i-secondary i-r
                , author i-secondary i-r
                Iprefetti title i-secondary i-r
                dell title i-secondary i-r
                ’ title i-secondary i-r
                Italia title i-secondary i-r
                napoleonica title i-secondary i-r
                , title i-secondary i-r
                Bologna publicationplace i-secondary i-r
                1983 year e-secondary i-r
                . year e-secondary e-r

        :param filepath: Path to the data
        :return: Four arrays: The first one contains sentences (one array of words per sentence) and the other threes are arrays of tags.

    """

    # Arrays to return
    words = []
    tags_1 = []
    tags_2 = []
    tags_3 = []

    word = tags1 = tags2 = tags3 = []
    with open (filepath, "r") as file:
        for line in file:
            if 'DOCSTART' not in line: #Do not take the first line into consideration
                # Check if empty line
                if line in ['\n', '\r\n']:
                    # Append line
                    words.append(word)
                    tags_1.append(tags1)
                    tags_2.append(tags2)
                    tags_3.append(tags3)

                    # Reset
                    word = []
                    tags1 = []
                    tags2 = []
                    tags3 = []

                else:
                    # Split the line into words, tag #1, tag #2, tag #3
                    w = line[:-1].split(" ")
                    word.append(w[0])
                    tags1.append(w[1])
                    tags2.append(w[2])
                    tags3.append(w[3])

    return words,tags_1,tags_2,tags_3




def setPrintToFile(filename):
    """
        Redirect all prints into a file

        :param filename: File to redirect all prints
        :return: the file and the original print "direction"
    """

    # Retrieve current print direction
    stdout_original = sys.stdout
    # Create file
    f = open(filename, 'w')
    # Set the new print redirection
    sys.stdout = f
    return f,stdout_original
    

def closePrintToFile(f, stdout_original):
    """
        Change the print direction and closes a file.

        :param filename: File to close
        :param stdout_original: Print direction
    """
    sys.stdout = stdout_original
    f.close()




def mergeDigits(datas, digits_word):
    """
        All digits in the given data will be mapped to the same word

        :param datas: The data to transform
        :param digits_word: Word to map digits to
        :return: The data transformed data
    """
    return [[[digits_word if x.isdigit() else x for x in w ] for w in data] for data in datas]



def indexData_x(x, ukn_words):
    """
        Map each word in the given data to a unique integer. A special index will be kept for "out-of-vocabulary" words.

        :param x: The data
        :return: Two dictionaries: one where words are keys and indexes values, another one "reversed" (keys->index, values->words)
    """

    # Retrieve all words used in the data (with duplicates)
    all_text = [w for e in x for w in e]
    # Compute the unique words (remove duplicates)
    words = list(set(all_text))
    print("Number of  entries: ",len(all_text))
    print("Individual entries: ",len(words))

    # Assign an integer index for each individual word
    word2ind = {word: index for index, word in enumerate(words, 2)}
    ind2word = {index: word for index, word in enumerate(words, 2)}

    # To deal with out-of-vocabulary words
    word2ind.update({ukn_words:1})
    ind2word.update({1:ukn_words})

    # The index '0' is kept free in both dictionaries

    return word2ind, ind2word


def indexData_y(y):
    """
        Map each word in the given data to a unique integer.

        :param y: The data
        :return: Two dictionaries: one where words are keys and indexes values, another one "reversed" (keys->index, values->words)
    """
    
    # Unique attributes in the data, sort alphabetically
    labels_t1 = list(set([w for e in y for w in e]))
    labels_t1 = sorted(labels_t1, key=str.lower)
    print("Number of labels: ", len(labels_t1))
    
    # Assign an integer index for each individual label
    label2ind = {label: index for index, label in enumerate(labels_t1, 1)}
    ind2label = {index: label for index, label in enumerate(labels_t1, 1)}
    
    # The index '0' is kept free in both dictionaries

    return label2ind, ind2label


def encodePadData_x(x, word2ind, maxlen, ukn_words, padding_style):
    """
        Transform a data of words in a data of integers, where each entrie as the same length.

        :param x: The data to transform
        :param word2ind: Dictionary to retrieve the integer for each word in the data
        :param maxlen: The length of each entry in the returned data
        :param ukn_words: Key, in the dictionary words-index, to use for words not present in the dictionary
        :param padding_style: Padding style to use for having each entry in the data with the same length
        :return: The tranformed data
    """
    print ('Maximum sequence length - general :', maxlen)
    print ('Maximum sequence length - data    :', max([len(xx) for xx in x]))
    
    # Encode: Map each words to the corresponding integer
    X_enc = [[word2ind[c] if c in word2ind.keys() else word2ind[ukn_words] for c in xx ] for xx in x]

    # Pad: Each entry in the data must have the same length
    X_encode = pad_sequences(X_enc, maxlen=maxlen, padding=padding_style)

    return X_encode


def encodePadData_y(y, label2ind, maxlen, padding_style):
    """
        Apply one-hot-encoding to each label in the dataset. Each entrie will have the same length

        Example:
            Input:  label2ind={Label_A:1, Label_B:2, Label_C:3}, maxlen=4
                    y=[ [Label_A, Label_C]                     ,     [Label_A, Label_B, Label_C] ]
            Output:   [ [[1,0,0], [0,0,1], [0,0,0], , [0,0,0]] ,     [[1,0,0], [0,1,0], [0,0,1]], [0,0,0],  ]

        :param y: The data to encode
        :param label2ind:  Dictionary where each value in the data is mapped to a unique integer
        :param maxlen: The length of each entry in the returned data
        :param padding_style: Padding style to use for having each entry in the data with the same length
        :return: The transformed data
    """

    print ('Maximum sequence length - labels :', maxlen)
    
    # Encode y (with pad)
    def encode(x, n):
        """
            Return an array of zeros, except for an entry set to 1 (one-hot-encode)
            :param x: Index entry to set to 1
            :param n: Length of the array to return
            :return: The created array
        """
        result = np.zeros(n)
        result[x] = 1
        return result

    # Transform each label into its index in the data
    y_pad = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]
    # One-hot-encode label
    max_label = max(label2ind.values()) + 1
    y_enc = [[encode(c, max_label) for c in ey] for ey in y_pad]
    
    # Repad (to have numpy array)
    y_encode = pad_sequences(y_enc, maxlen=maxlen, padding=padding_style)
    
    return y_encode



def characterLevelIndex(X, digits_word):
    """
        Map each character present in the dataset into an unique integer. All digits are mapped into a single array.

        :param X: Data to retrieve characters from
        :param digits_word: Words regrouping all digits
        :return: A dictionary where each character is maped into a unique integer, the maximum number of words in the data, the maximum of characters in a word
    """

    # Create a set of all character
    all_chars = list(set([c for s in X for w in s for c in w]))

    # Create an index for each character
    # The index 1 is reserved for the digits, regrouped under the word param `digits_word`
    char2ind = {char: index for index, char in enumerate(all_chars, 2)}
    ind2char = {index: char for index, char in enumerate(all_chars, 2)}

    # To deal with out-of-vocabulary words
    char2ind.update({digits_word:1})
    ind2char.update({1:digits_word})

    # For padding
    maxWords = max([len(s) for s in X])
    maxChar  = max([len(w) for s in X for w in s])
    print("Maximum number of words in a sequence  :", maxWords)
    print("Maximum number of characters in a word :", maxChar)

    return char2ind, maxWords, maxChar


def characterLevelData(X, char2ind, maxWords, maxChar, digits_word, padding_style):
    """
        For each word in the data, transform it into an array of characters. All characters array will have the same length. All sequence will have the same array length.
        All digits will be maped to the same character arry.
        If a character is not present in the dictionary character-index, discard it.

        :param X: The data
        :param chra2ind: Dictionary where each character is maped to a unique integer
        :param maxWords: Maximum number of words in a sequence
        :param maxChar: Maximum number of characters in a word
        :param digits_word: Word regrouping all digits. 
        :param padding_style: Padding style to use for having each entry in the data with the same length
        :return: The transformed array
    """

    # Transform each word into an array of characters (discards those oov)
    X_char = [[[char2ind[c] for c in w if c in char2ind.keys()] if w!=digits_word else [1] for w in s] for s in X]

    # Pad words - Each words has the same number of characters
    X_char = pad_sequences([pad_sequences(s, maxChar, padding=padding_style) for s in X_char], maxWords, padding=padding_style)
    return X_char
    


def word2VecEmbeddings(word2ind, num_features_embedding):
    """
        Convert a file of pre-computed word embeddings into dictionary: {word -> embedding vector}. Only return words of interest. 
        If the word isn't in the embedding, returned a zero-vector instead.

        :param word2ind: Dictionary {words -> index}. The keys represented the words for each embeddings will be retrieved.
        :param num_features_embedding: Size of the embedding vectors
        :return: Array of embeddings vectors. The embeddings vector at position i corresponds to the word with value i in the dictionary param `word2ind`
    """

    # Pre-trained embeddings filepath
    file_path = "dataset/pretrained_vectors/vecs_{0}.txt".format(num_features_embedding)
    ukn_index = "$UKN$"

    # Read the embeddings file
    embeddings_all = {}
    with open (file_path, "r") as file:
        for line in file:
            l = line.split(' ')
            embeddings_all[l[0]] = l[1:]

    # Compute the embedding for each word in the dataset
    embedding_matrix = np.zeros((len(word2ind)+1, num_features_embedding))
    for word, i in word2ind.items():
        if word in embeddings_all:
            embedding_matrix[i] = embeddings_all[word]
#        else:
 #           embedding_matrix[i] = embeddings_all[ukn_index]

    # Delete the word2vec dictionary from memory
    del embeddings_all

    return embedding_matrix



class Classification_Scores(Callback):
    """
        Add the F1 score on the testing data at the end of each epoch.
        In case of multi-outputs, compute the F1 score for each output layer and the mean of all F1 scores.
        Compute the training F1 score for each epoch. Store the results internally.
        Internally, the accuracy and recall scores will also be stored, both for training and testing dataset.
        The model's weigths for the best epoch will be save in a given folder.
    """
    
    def __init__(self, train_data, ind2label, model_save_path):
        """
            :param train_data: The data used to compute training accuracy. One array of two arrays => [X_train, y_train]
            :param ind2label: Dictionary of index-label to add tags label into results
            :param model_save_path: Path to save the best model's weigths
        """
        self.train_data = train_data
        self.ind2label = ind2label
        self.model_save_path = model_save_path
        self.score_name = 'val_f1'

    

    def on_train_begin(self, logs={}):
        self.test_report = []
        self.test_f1s = []
        self.test_acc = []
        self.test_recall = []
        self.train_f1s = []
        self.train_acc = []
        self.train_recall = []

        self.best_score = -1
        
        # Add F1-score as a metric to print at end of each epoch
        self.params['metrics'].append("val_f1")
        
        # In case of multiple outputs
        if len(self.model.layers) > 1:
            for output_layer in self.model.layers:
                self.params['metrics'].append("val_"+output_layer.name+"_f1")


                
    def compute_scores(self, pred, targ):
        """
            Compute the Accuracy, Recall and F1 scores between the two given arrays pred and targ (targ is the golden truth)
        """
        val_predict = np.argmax(pred, axis=-1)
        val_targ = np.argmax(targ, axis=-1)

        # Flatten arrays for sklearn
        predict_flat = np.ravel(val_predict)
        targ_flat = np.ravel(val_targ)

        # Compute scores
        return precision_recall_fscore_support(targ_flat, predict_flat, average='weighted', labels=[x for x in np.unique(targ_flat) if x!=0])[:3]

    
    def compute_epoch_training_F1(self):
        """
            Compute and save the F1 score for the training data
        """
        in_length  = len(self.model._input_layers)
        out_length = len(self.model.layers)
        predictions = self.model.predict(self.train_data[0])
        if len(predictions) != out_length:
            predictions = [predictions]
    
        vals_acc = []
        vals_recall = []
        vals_f1 = []
        for i,pred in enumerate(predictions):
            _val_acc, _val_recall, _val_f1 = self.compute_scores(np.asarray(pred), self.train_data[1][i])
            vals_acc.append(_val_acc)
            vals_recall.append(_val_recall)
            vals_f1.append(_val_f1)
        
        self.train_acc.append(sum(vals_acc)/len(vals_acc))
        self.train_recall.append(sum(vals_recall)/len(vals_recall))
        self.train_f1s.append(sum(vals_f1)/len(vals_f1))
    
    
    def classification_report(self, i, pred, targ, printPadding=False):
        """
            Comput the classification report for the predictions given.
        """

        # Hold all classification reports
        reports = []
        
        # The model predicts probabilities for each tag. Retrieve the id of the most probable tag.
        pred_index = np.argmax(pred, axis=-1)
        # Reverse the one-hot encoding for target
        true_index = np.argmax(targ, axis=-1) 

        # Index 0 in the predictions referes to padding
        ind2labelNew = self.ind2label[i].copy()
        ind2labelNew.update({0: "null"})

        # Compute the labels for each prediction
        pred_label = [[ind2labelNew[x] for x in a] for a in pred_index]
        true_label = [[ind2labelNew[x] for x in b] for b in true_index]

        # CLASSIFICATION REPORTS 
        reports.append("")
        if printPadding:
            reports.append("With padding into account")
            reports.append(metrics.flat_classification_report(true_label, pred_label, digits=4))
            reports.append("")
            reports.append('----------------------------------------------')
            reports.append("")
            reports.append("Without the padding:")
        reports.append(metrics.flat_classification_report(true_label, pred_label, digits=4, labels=list(self.ind2label[i].values())))
        return '\n'.join(reports)
    

    def on_epoch_end(self, epoch, logs={}):
        """
            At the end of each epoch, compute the F1 score for the validation data.
            In case of multi-outputs model, compute one value per output and average all to return the overall F1 score.
            Same model's weights for the best epoch.
        """
        self.compute_epoch_training_F1()
        in_length  = len(self.model._input_layers)  # X data - to predict from
        out_length = len(self.model.layers) # Number of tasks
        
        # Compute the model predictions
        predictions = self.model.predict(self.validation_data[:in_length])
        # In case of single output
        if len(predictions) != out_length:
            predictions = [predictions]
        
        
        vals_acc = []
        vals_recall = []
        vals_f1 = []
        reports = ""
        # Iterate over all output predictions
        for i,pred in enumerate(predictions):
            _val_acc, _val_recall, _val_f1 = self.compute_scores(np.asarray(pred), self.validation_data[in_length+i])

            # Classification report
            reports += "For task "+str(i+1)+"\n"
            reports += "===================================================================================="
            reports += self.classification_report(i,np.asarray(pred), self.validation_data[in_length+i]) + "\n\n\n"
            
            # Add scores internally
            vals_acc.append(_val_acc)
            vals_recall.append(_val_recall)
            vals_f1.append(_val_f1)
            
            # Add F1 score to be log
            f1_name = "val_"+self.model.layers[i].name+"_f1"
            logs[f1_name] = _val_f1
            

        # Add classification reports for all the predicitions/tasks
        self.test_report.append(reports)

        # Add internally
        self.test_acc.append(sum(vals_acc)/len(vals_acc))
        self.test_recall.append(sum(vals_recall)/len(vals_recall))
        self.test_f1s.append(sum(vals_f1)/len(vals_f1))
        
        # Add to log
        f1_mean = sum(vals_f1)/len(vals_f1)
        logs["val_f1"] = f1_mean

        # Save best model's weights
        if f1_mean > self.best_score:
            self.best_score = f1_mean
            save_load_utils.save_all_weights(self.model, self.model_save_path)



def write_to_csv(filename, columns, rows):
    """
        Create a .csv file with the data given

        :param filename: Path and name of the .csv file, without csv extension
        :param columns: Columns of the csv file (First row of the file)
        :param rows: Data to write into the csv file, given per row

    """
    with open(filename+'.csv', 'w') as csvfile:
        wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        wr.writerow(columns)    
        for n in rows:
            wr.writerow(n)


def save_model_training_scores(filename, hist, classification_scores):
    """
        Create a .csv file containg the model training metrics for each epoch

        :param filename: Path and name of the .csv file without csv extension
        :param hist: Default model training history returned by Keras
        :param classification_scores: Classification_Scores instance used as callback in the model's training 

        :return: Nothing.
    """
    csv_values = []

    csv_columns = ["Epoch", "Training Accuracy", "Training Recall", "Training F1", "Testing Accuracy", "Testing Recall", "Testing F1"]

    csv_values.append(hist.epoch) # Epoch column

    # Training metrics
    csv_values.append(classification_scores.train_acc)     # Training Accuracy column
    csv_values.append(classification_scores.train_recall)  # Training Recall column
    csv_values.append(classification_scores.train_f1s)     # Training F1 column

    # Testing metrics
    csv_values.append(classification_scores.test_acc)       # Testing Accuracy column
    csv_values.append(classification_scores.test_recall)    # Testing Accuracy column 
    csv_values.append(classification_scores.test_f1s)       # Testing Accuracy column

    # Creste file
    write_to_csv(filename, csv_columns, zip(*csv_values))
    return


def model_best_scores(classification_scores, best_epoch):
    """
        Return the metrics from best epoch

        :param classification_scores: Classification_Scores instance used as callback in the model's training 
        :param best_epoch: Best training epoch index

        :return Best epoch training metrics: ["Best epoch", "Training Accuracy", "Training Recall", "Training F1", "Testing Accuracy", "Testing Recall", "Testing F1"]
    """
    best_values = []
    best_values.append(1 + best_epoch)

    best_values.append(classification_scores.train_acc[best_epoch])
    best_values.append(classification_scores.train_recall[best_epoch])
    best_values.append(classification_scores.train_f1s[best_epoch])

    best_values.append(classification_scores.test_acc[best_epoch])
    best_values.append(classification_scores.test_recall[best_epoch])
    best_values.append(classification_scores.test_f1s[best_epoch])

    return best_values



def compute_predictions(model, X, y, ind2label, nbrTask=-1):
    """
        Compute the predictions and ground truth

        :param model: The model making predictions
        :param X: Data
        :param y: Ground truth
        :param ind2label: Dictionaries of index to labels. Used to return have labels to predictions.

        :return: The predictions and groud truth ready to be compared, flatten (1-d array).
    """

    # Compute training score
    pred = model.predict(X)
    if len(model.outputs)>1: # For multi-task
        pred = pred[nbrTask]
    pred = np.asarray(pred)
    # Compute validation score
    pred_index = np.argmax(pred, axis=-1)

    # Reverse the one-hot encoding
    true_index = np.argmax(y, axis=-1) 

    # Index 0 in the predictions referes to padding
    ind2labelNew = ind2label.copy()
    ind2labelNew.update({0: "null"})
    
    # Compute the labels for each prediction
    pred_label = [[ind2labelNew[x] for x in a] for a in pred_index]
    true_label = [[ind2labelNew[x] for x in b] for b in true_index]

    # Flatten data
    predict_flat = np.ravel(pred_label)
    targ_flat = np.ravel(true_label)

    return predict_flat, targ_flat



def save_confusion_matrix(y_target, y_predictions, labels, figure_path, figure_size=(20,20)):
    """
        Generate two confusion matrices plots: with and without normalization.

        :param y_target: Tags groud truth
        :param y_predictions: Tags predictions
        :param labels: Predictions classes to use
        :param figure_path: Path the save figures
        :param figure_size: Size of the generated figures

        :return: Nothing
    """

    # Compute confusion matrices
    cnf_matrix = confusion_matrix(y_target, y_predictions)

    # Confusion matrix 
    plt.figure(figsize=figure_size)
    plot_confusion_matrix(cnf_matrix, classes=labels, title='Confusion matrix, without normalization')
    plt.savefig("{0}.png".format(figure_path))
    
    # Confusion matrix  with normalization
    plt.figure(figsize=figure_size)
    plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True, title='Normalized confusion matrix')
    plt.savefig("{0}_normalized.png".format(figure_path))

    return


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, printToFile=False):
    """
        FROM: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if printToFile: print("Normalized confusion matrix")
    else:
        if printToFile: print('Confusion matrix, without normalization')

    if printToFile: print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
