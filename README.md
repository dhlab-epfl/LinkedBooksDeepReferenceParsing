# LinkedBooksDeepReferenceParsing

This repository contains the code for the following article:
    
    @article{alves_deep_2018,
          author       = {{Rodrigues Alves, Danny and Giovanni Colavizza and Frédéric Kaplan}},
          title        = {{Deep Reference Mining from Scholarly Literature in the Arts and Humanities}},
          journal      = {{Submitted to Frontiers in Research Metrics & Analytics}},
          year         = 2018
        }

## TODO

Giovanni

*	~~Add dataset to repo and update this file (remove the not used annotation scheme, position 3 in a split)~~
*	~~How to share pretrained vectors? is it worth it? Zenodo is an option.~~
*   Paper writing
*   ~~Tensorflow code~~
*   Check everything

Danny

*   ~~Create high-res (vector) figures (and fix the fontsize of some of them!)~~
*   ~~Check paper as I write, I am adding small TODOs there!~~
*   ~~Prepare a data analysis notebook with a selection of results (esp. from 0. Data Analysis, Appendix C)~~
*   ~~Push (in a separate branch) the clean Keras code (for single and multi task)~~
*   ~~Add info on how to use it in the README~~
*   Provide some more on error analysis (examples of miss-classified instances, etc.). TO BE DISCUSSED first
*   Code cleanup
*   Tests with learning rate and multi-task fine-tune
*   Improve Keras readme ?
*   ~~Neural ParsCit~~

## Task

## Contents

* `LICENSE` MIT.
* `README.md` this file.
* `dataset/`
    * [annotated_dataset](dataset/annotated_dataset.json.zip) The annotated dataset in json format (zip compressed).
    * [report.p](dataset/report.p) A set of statistics on the annotated dataset, pickled.
    * [report.txt](dataset/report.txt) A set of statistics on the annotated dataset, txt.
    * [sources](dataset/sources.csv) List of monographs and journal issues which have been annotated, with number of annotated pages for each, and link to the Italian catalog for the relative entry. Separator `,`.
* `M1.ipynb` a Python notebook to train a CRF parsing model using specific reference tags (e.g. author, title, publication year). You can use the annotated dataset in json format as input here.
* `M2.ipynb` a Python notebook to train a CRF parsing model using generic begin/end reference tags (e.g. begin-secondary, in-secondary, end-secondary for a reference to a secondary source).
* `models/`
    * [modelM1](models/modelM1_ALL_L.pkl) trained model 1, details in the paper.
    * [modelM2](models/modelM1_ALL_L.pkl) trained model 2, details in the paper.
* `code/`
    * [support_functions](code/support_functions.py) supporting functions for training/testing, plotting and parsing references.
    * [feature_extraction](code/feature_extraction_words.py) feature extraction functions, document level.
    * [feature_extraction_supporting_functions](code/feature_extraction_supporting_functions_words.py) feature extraction functions, token level.


## Dataset

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1175213.svg)](https://doi.org/10.5281/zenodo.1175213)

## Implementations

### CRF baseline

### Keras

The directory contains code to run the models with a Keras implementation. Code for both single and multitask models are given. For the single tasks, one model for each of the three tasks will be computed by running the python script *main_threeTasks.py*. The multitask learning model can be computed with the script *main_multiTaskLearning.py*.

The data is expected to be in a *data* folder, one directory up, with three files inside it: *train.txt* for the training dataset, *test.txt* for the testing dataset, and *valid.txt* for the validation dataset.

The results will be stored into the *model_results* folder, with one directory created for each model.

### Tensor Flow

See internal [readme](tensorflow/README.md) for details.

This implementation borrows from [Guillaume Genthial's Sequence Tagging with Tensorflow](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html).







