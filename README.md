# Deep Reference Parsing

This repository contains the code for the following article:
    
    @article{alves_deep_2018,
          author       = {{Rodrigues Alves, Danny and Giovanni Colavizza and Frédéric Kaplan}},
          title        = {{Deep Reference Mining from Scholarly Literature in the Arts and Humanities}},
          journal      = {{Frontiers in Research Metrics & Analytics}},
          volume       = 3,
          number       = 21,
          year         = 2018,
          doi          = {10.3389/frma.2018.00021}
        }

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
* [crf_baseline](crf_baseline) CRF baseline implementation details.
* [keras](keras) Keras implementation details.
* [tensorflow](tensorflow) TF implementation details.

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

