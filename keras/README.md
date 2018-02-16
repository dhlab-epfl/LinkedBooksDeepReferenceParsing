## Keras

The directory contains code to run the models with a Keras implementation and a Tensorlow backend. Code for both single and multitask models are given. For the single tasks, one model for each of the three tasks will be computed by running the python script *main_threeTasks.py*. The multitask learning model can be computed with the script *main_multiTaskLearning.py*.

The data is expected to be in a *data* folder, one directory up, with three files inside it: *train.txt* for the training dataset, *test.txt* for the testing dataset, and *valid.txt* for the validation dataset.

The results will be stored into the *model_results* folder, with one directory created for each model.

## Import
The experiments where produced with:
* Keras : version 2.1.1
* TensorFlow: 1.4.0
* Numpy: 1.13.3
* [Keras contrib](https://github.com/keras-team/keras-contrib) Keras contrib : 0.0.2
* Sklearn : 0.19.1
* [Sklearn crfsuite](https://sklearn-crfsuite.readthedocs.io/en/latest/index.html) Sklearn crfsuite : 0.3.6
* Python 3.5