# CRF baseline implementation

## How to
The directory contains code to run the CRF model used as baseline. Code to train, fine-tune and validate models are given. 

For single tasks, one model for each of the three tasks will be computed by running the python script *main_threeTasks.py*: each model will be stored under the folder *models*. In order to fine-tune the two CRF model parameters *c1* and *c2*, the python script *main_finetune.py* can be run: a plot gathering the results is saved in the folder *plots* and the best model will be save in the *models* folder. To compute the validation score, the script *validation.py* needs to have the models generated previously *crf_t1.pkl*, *crf_t2.pkl* and *crf_t3.pkl* and will print the classification scores.

The data is expected to be in a *dataset* folder, in the main repository directory, with three files inside it: *clean_train.txt* for the training dataset, *clean_test.txt* for the testing dataset, and *clean_valid.txt* for the validation dataset.

    python main_finetune.py
    python main_threeTasks.py
    python validation.py


## Contents
* `README.md` this file.
* `code/`
    * [feature_extraction_supporting_functions_words](code/feature_extraction_supporting_functions_words.py) helper functions to extract features from words.
    * [feature_extraction_words](code/feature_extraction_words.py) functions to extract features from words.
    * [utils](code/utils.py) utility functions to load data and redirected log files.
* [main_finetune](main_finetune.py) python script to fine-tune two model parameters.
* [main_threeTasks](main_threeTasks.py) python script to train one CRF model for each task.
* [validation](validation.py) python script to compute classification score on validation dataset for the three tasks.

## Dependencies 
* Numpy: 1.13.3
* Sklearn : 0.19.1
* [Sklearn crfsuite](https://sklearn-crfsuite.readthedocs.io/en/latest/index.html) Sklearn crfsuite : 0.3.6
* Python 3.5