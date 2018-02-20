"""
Use a pretrained model for predictions
Borrows from: https://github.com/guillaumegenthial/sequence_tagging
"""

import os

from ref_model import RefModel
from model.data_utils import load_vocab, get_processing_word, coNLLDataset_full

def interactive_shell(model):
    """Creates interactive shell to play with model

    :param model: instance of RefModel
    """
    print("""
    This is an interactive mode.
    To exit, enter 'exit'.
    You can enter a sentence like
    input> I love Paris""")

    while True:
        sentence = input("input> ")

        words_raw = sentence.strip().split()

        if words_raw in ["exit","quit","bye","q","stop"]:
            break

        preds = model.predict(words_raw)

        print(" ".join(words_raw))
        print(" ".join(preds))

# dataset locations and basic configs
filename_dev = "../dataset/clean_test.txt"
filename_test = "../dataset/clean_valid.txt"
filename_train = "../dataset/clean_train.txt"
which_tags = -3  # -1, -2, -3: Ackerman author b-secondary b-r
use_chars = True
max_iter = None  # if None, max number of examples in Dataset

# general config
dir_output = "results/test_run"
dir_model = os.path.join(dir_output, "model.weights") # not in use here, best model is stored in primary memory

# vocabs (created with build_data)
filename_words = "working_dir/words.txt"
filename_words_ext = "working_dir/words_ext.txt"
filename_tags = "working_dir/tags.txt"
filename_chars = "working_dir/chars.txt"

# load vocabs
vocab_words = load_vocab(filename_words)
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
model = RefModel(processing_word=processing_word,processing_tag=processing_tag,vocab_chars=vocab_chars,
                 vocab_words=vocab_words,vocab_tags=vocab_tags,nwords=nwords,nchars=nchars,
                 ntags=ntags,dir_output=dir_output,dir_model=dir_model,use_chars=use_chars,random_state=0,
                 use_pretrained=True, hidden_size_char=50, batch_size=100, lr_decay=1, l2_reg_lambda=0,
                 use_crf=True, use_cnn=False, dim_word=300, hidden_size_lstm=200, lr=0.001,
                 train_embeddings=True, dim_char=100, lr_method="rmsprop")

model.restore_session()
interactive_shell(model)