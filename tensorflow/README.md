# Tensorflow implementation

## How to
A model can be trained using ref_model. All parameters can be tuned there too (see comments). In order to do this, be sure to have both dataset and pretrained vectors in the main directory, or change folder info accordingly.

    python ref_model.py
    
Once a model is trained, it can be used interactively calling `play_with.py`. Model selection can be done using `cv_model.py` which implements grid search.

## Contents
* `README.md` this file.
* `utils/`
    * [data utils](utils/data_utils.py) general data utility functions.
    * [general utils](utils/general_utils.py) other utility functions.
* [reference parsing model](ref_model.py) contains the main RefModel model discussed in the paper, and can be run to train an instance (assumes the dataset and pretrained vectors are available).
* [cross validation](cv_model.py) contains code to fot multiple models for model selection or fine tuning (assumes the dataset and pretrained vectors are available).
* [play with](play_with.py) contains code to load a model and use it with an interactive terminal.

## Dependencies 
* TensorFlow: 1.4.0
* Numpy: 1.13.3
* Sklearn : 0.19.1
* Python 3.5

## Future work
* Add a conf file, ideally shared with the implementation in Keras.
* Add a multitask implementation.