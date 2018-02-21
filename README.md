# Deep Reference Parsing

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
*   ~~Code cleanup~~
*   Test learning rate and multi-task fine-tune
*   Improve Keras readme 
*   ~~add CRF baseline code in crf_baseline/~~
*   ~~Update notebook with new data organisation~~
*   ~~Neural ParsCit~~

## Task definition

We focus on the task of reference mining, instantiated into three tasks: reference components detection (task 1), reference typology detection (task 2) and reference span detection (task 3).

* Sequence: *G. Ostrogorsky, History of the Byzantine State, Rutgers University Press, 1986.*
* Task 1: *author author title title title title title publisher publisher publisher year*
* Task 2: *b-secondary i-secondary ... e-secondary*
* Task 3: *b-r i-r ... e-r*

## Contents

* `LICENSE` MIT.
* `README.md` this file.
* `dataset/`
    * [train](dataset/clean_test.txt) Train split, CoNLL format.
    * [test](dataset/clean_train.txt) Test split, CoNLL format.
    * [validation](dataset/clean_valid.txt) Validation split, CoNLL format.
* [compressed dataset](dataset.tar.gz) Compressed dataset.
* [data facts](Data%20Facts.ipynb) a Python notebook to explore the dataset (number of references, tag distributions).
* `crf_baseline/`
    * [readme](crf_baseline/README.md) CRF baseline implementation details.
    * ...
* `keras/`
    * [readme](keras/README.md) Keras implementation details.
    * ...
* `tensorflow/`
    * [readme](tensorflow/README.md) TF implementation details.
    * ...

## Dataset

Example of dataset entry (beginning of validation dataset, first line/sequence): Token Task1tag Task2tag Task3tag`:

    -DOCSTART- -X- -X- o


    C author b-secondary b-r
    . author i-secondary i-r
    Agnoletti author i-secondary i-r
    , author i-secondary i-r
    Treviso title i-secondary i-r
    e title i-secondary i-r
    le title i-secondary i-r
    sue title i-secondary i-r
    pievi title i-secondary i-r
    . title i-secondary i-r
    Illustrazione title i-secondary i-r
    storica title i-secondary i-r
    , title i-secondary i-r
    Treviso publicationplace i-secondary i-r
    1898 year i-secondary i-r
    , year i-secondary i-r
    2 publicationspecifications i-secondary i-r
    v publicationspecifications e-secondary i-r
    . publicationspecifications e-secondary e-r

Pre-trained word vectors can be downloaded from Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1175213.svg)](https://doi.org/10.5281/zenodo.1175213)

## Implementations

### CRF baseline

See internal [readme](crf_baseline/README.md) for details.

### Keras

See internal [readme](keras/README.md) for details.

### Tensor Flow

See internal [readme](tensorflow/README.md) for details.

This implementation borrows from [Guillaume Genthial's Sequence Tagging with Tensorflow](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html).

## Please cite as

TBD
