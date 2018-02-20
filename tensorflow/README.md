# Tensorflow implementation

This implementation borrows from [Guillaume Genthial's Sequence Tagging with Tensorflow](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html).

## How to
A model can be trained using ref_model. All parameters can be tuned there too (see comments). In order to do this, be sure to have both dataset and pretrained vectors in the main directory, or change folder info accordingly.

## Contents
* `README.md` this file.
* `model/`
    * [data utils](model/data_utils.py) general data utility functions.
    * [general utils](model/general_utils.py) other utility functions.
* `ref_model.py` contains the main RefModel model discussed in the paper, and can be run to train an instance (assumes the dataset and pretrained vectors are available).
* `cv_model.py` contains code to fot multiple models for model selection or fine tuning (assumes the dataset and pretrained vectors are available).
* `play_with.py` contains code to load a model and use it with an interactive terminal.

## Dependencies 
* TensorFlow: 1.4.0
* Numpy: 1.13.3
* Sklearn : 0.19.1
* Python 3.5

## TODO
* add a conf file, ideally shared with the implementation in Keras