"""
Cross validation for model selection
"""

import numpy as np
import itertools as it
from collections import OrderedDict
import os
from model.data_utils import build_data, load_vocab, get_processing_word,\
        coNLLDataset_full
from ref_model import RefModel

# GLOBALS

# dataset locations and basic configs
filename_dev = "../dataset/clean_test.txt"
filename_test = "../dataset/clean_valid.txt"
filename_train = "../dataset/clean_train.txt"
which_tags = -3  # -1, -2, -3: Ackerman author b-s b-secondary b-r
task_dir = "cv_%d"%which_tags
use_chars = True # parameter to change globally
use_pretrained = True # parameter to change globally
max_iter = None  # if not None, max number of examples in Dataset
n_epocs = 25
dim_words = [100,300] # pretrained word embeddings, be they exist!

# vocabs (created with build_data)
filename_words = "working_dir/words.txt"
filename_words_ext = "working_dir/words_ext.txt"
filename_tags = "working_dir/tags.txt"
filename_chars = "working_dir/chars.txt"

# build data for all possible models
build_data(filename_dev,filename_test,filename_train,dim_words,filename_words,
               filename_words_ext,filename_tags,filename_chars,
               filename_word="../pretrained_vectors/vecs_{}.txt",
               filename_word_vec_trimmed="../pretrained_vectors/vecs_{}.trimmed.npz",
               which_tags=which_tags)

# load vocabs
vocab_words = load_vocab(filename_words)
if use_pretrained:
    vocab_words = load_vocab(filename_words_ext)
vocab_tags = load_vocab(filename_tags)
vocab_chars = load_vocab(filename_chars)
nwords = len(vocab_words)
nchars = len(vocab_chars)
ntags = len(vocab_tags)

# load data
processing_word = get_processing_word(vocab_words,
                                      vocab_chars, lowercase=True, chars=use_chars)
processing_tag = get_processing_word(vocab_tags,
                                     lowercase=False, allow_unk=False)
X_dev, y_dev = coNLLDataset_full(filename_dev, processing_word, processing_tag, max_iter, which_tags)
X_train, y_train = coNLLDataset_full(filename_train, processing_word, processing_tag, max_iter, which_tags)
X_valid, y_valid = coNLLDataset_full(filename_test, processing_word, processing_tag, max_iter, which_tags)

print("Size of train, test and valid sets (in number of sentences): ")
print(len(X_train), " ", len(y_train), " ", len(X_dev), " ", len(y_dev), " ", len(X_valid), " ", len(y_valid))

def train_model(config,conf_id):
    """Train, evaluates and reports on a single model

    :param config: (dict) parameter configuration
    :param conf_id: (int) id of the configuration to fit
    :return: None
    """

    # general config
    model_name = str(config)
    print("Model configuration:",model_name)
    dir_output = "results/%s/%s_%s_%d"%(task_dir,str(use_pretrained),str(use_chars),conf_id)
    print("Model directory:", dir_output)
    os.makedirs(dir_output, exist_ok=True)
    os.makedirs(dir_output, exist_ok=True)
    with open(os.path.join(dir_output, "config_%s_%s_%d.txt"%(str(use_pretrained),str(use_chars),c)), "w") as f:
        f.write(model_name)
    dir_model = os.path.join(dir_output, "model.weights")

    model = RefModel(processing_word=processing_word, processing_tag=processing_tag, vocab_chars=vocab_chars,
                     vocab_words=vocab_words, vocab_tags=vocab_tags, nwords=nwords, nchars=nchars,
                     ntags=ntags, dir_output=dir_output, dir_model=dir_model, dim_word=config["dim_word"],dim_char=config["dim_char"],
                     use_pretrained=config["use_pretrained"],train_embeddings=config["train_embeddings"],
                     dropout=config["dropout"],batch_size=config["batch_size"],lr_method=config["lr_method"],lr=config["lr"],
                     lr_decay=config["lr_decay"],clip=config["clip"],nepoch_no_imprv=config["nepoch_no_imprv"],l2_reg_lambda=config["l2_reg_lambda"],
                     hidden_size_char=config["hidden_size_char"],hidden_size_lstm=config["hidden_size_lstm"],
                     use_crf=config["use_crf"],use_chars=config["use_chars"],use_cnn=config["use_cnn"],random_state=config["random_state"])

    # fit
    fitted = model.fit(X_train, y_train, X_dev, y_dev, n_epocs)
    print("Test final f1 score: ", fitted.best_score)
    ev_msg = fitted.evaluate(X_valid, y_valid)

    # report
    with open(os.path.join("results/%s"%(task_dir),"cv_report.txt"),"a") as f:
        f.write("------------\n")
        f.write("Model: %s\n"%model_name)

        f.write("Test final f1 score: %f\n"%fitted.best_score)

        # evaluate
        f.write("Evaluation: %s\n"%str(ev_msg))
    with open(os.path.join("results/%s"%(task_dir), "cv_report.csv"), "a") as f:
        f.write("Model_%s_%s_%d"%(str(use_pretrained),str(use_chars),c)+";"+str(fitted.best_score)+";"+str(ev_msg["f1"])+";"+str(ev_msg["acc"])+";"+str(ev_msg["p"])+";"+str(ev_msg["r"])+"\n")

if __name__ == "__main__":

    # Param search
    # NB use chars or not is decided above, as is the task (which_tags).
    param_distribs = OrderedDict({
        "dim_word"          : [100,300],
        "dim_char"          : [100,300],
        "use_pretrained"    : [use_pretrained], # see above
        "train_embeddings"  : [True,False], # only used if use_pretrained is True
        "dropout"           : [0.5],
        "batch_size"        : [50],
        "lr_method"         : ["adam"],
        "lr"                : [0.001],
        "lr_decay"          : [0.9],
        "clip"              : [-1],
        "nepoch_no_imprv"   : [5],
        "l2_reg_lambda"     : [0],
        "hidden_size_char"  : [100],
        "hidden_size_lstm"  : [300],
        "use_crf"           : [True,False],
        "use_chars"         : [use_chars], # see above
        "use_cnn"           : [True,False],
        "random_state"      : [0] # reproducibility
    })

    # create a list of configurations
    n_configs = np.prod([len(v) for v in param_distribs.values()])
    print("Total number of configurations to try:",n_configs)

    allNames = sorted(param_distribs)
    combinations = it.product(*(param_distribs[Name] for Name in allNames))
    combinations = list(combinations)
    assert len(combinations)==n_configs

    # initialize report csv file
    os.makedirs("results/%s"%(task_dir),exist_ok=True)
    if not os.path.isfile(os.path.join("results/%s"%(task_dir), "cv_report.csv")):
        with open(os.path.join("results/%s"%(task_dir), "cv_report.csv"), "w") as f:
            f.write("model_name;best_f1_test_score;f1_validation;accuracy_validation;precision_validation;recall_validation\n")

    for n,c in enumerate(combinations):
        config = {k:v for k,v in zip(allNames,c)}
        train_model(config,n)