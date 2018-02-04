# LinkedBooksDeepReferenceParsing

TBD

## TODO

Giovanni

*	Add dataset to repo and update this file
*	How to share pretrained vectors? is it worth it?
*   Paper writing
*   Tensorflow code
*   Check everything

Danny

*   Create high-res (vector) figures (and fix the fontsize of some of them!)
*   Prepare a data analysis notebook with a selection of results (esp. from 0. Data Analysis, Appendix C)
*   Push (in a separate branch) the clean Keras code (for single and multi task)
*   Add info on how to use it in the README
*   Provide some more on error analysis (examples of miss-classified instances, etc.). TO BE DISCUSSED first

## Task

## Dataset

## Implementations

### CRF baseline

### Keras

The Keras branch contains code to run the models. Code for both single and multitask are available. For the single tasks, one model for each of the three tasks will be computed by running the python script *main_threeTasks.py*. The multitask learning model can be computed with the script *main_multiTaskLearning.py*.

The data is expected to be in a *data* folder, with three files inside it: *train.txt* for the training dataset, *test.txt* for the testing dataset, and *valid.txt* for the validation dataset.

The results will be stored into the *model_results* folder.

### Tensor Flow