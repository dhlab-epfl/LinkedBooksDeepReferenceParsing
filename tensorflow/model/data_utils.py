"""
Utilities for dealing with data
Borrows from: https://github.com/guillaumegenthial/sequence_tagging
"""

import numpy as np
import tensorflow as tf

# shared global variables
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "o"

# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
        ERROR: Unable to locate file {}.
        
        FIX: Check that build_data has been called before training.
        """.format(filename)
        super(MyIOError, self).__init__(message)


def build_data(filename_dev,filename_test,filename_train,dim_words,filename_words,
               filename_words_ext,filename_tags,filename_chars,
               filename_word_vec="../pretrained_vectors/vecs_{}.txt",
               filename_word_vec_trimmed="../pretrained_vectors/vecs_{}.trimmed.npz",
               which_tags=-1):
    """
    Prepares the dataset before training a model.

    :param filename_dev: the file with test data (dev)
    :param filename_test: the file with validation data
    :param filename_train: the file with train data
    :param dim_words: dimensionality of word embeddings
    :param filename_words: filename where to put exported word vocabulary
    :param filename_words_ext: filename where to put exported word vocabulary
    :param filename_tags: filename where to put exported tag vocabulary
    :param filename_chars: filename where to put exported char vocabulary
    :param filename_word_vec: filename of word vectors
    :param filename_word_vec_trimmed: filename where to put exported trimmed word vectors
    :param which_tags: which tagging scheme to use (-1 -2 -3 or 3 2 1 for task 1 2 3 respectively)
    :return: None
    """

    processing_word = get_processing_word(lowercase=True)

    # Generators
    dev = CoNLLDataset(filename_dev, processing_word, which_tags=which_tags)
    test = CoNLLDataset(filename_test, processing_word, which_tags=which_tags)
    train = CoNLLDataset(filename_train, processing_word, which_tags=which_tags)

    # Build Word, Char and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])
    vocab = vocab_words
    vocab.add(UNK)
    vocab.add(NUM)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab, filename_words)
    write_vocab(vocab_tags, filename_tags)
    write_vocab(vocab_chars, filename_chars)

    # Export extended vocab
    vocab_vec = get_vec_vocab(filename_word_vec.format(dim_words[0])) # pick any, words are the same
    vocab = vocab & vocab_vec
    write_vocab(vocab, filename_words_ext)

    # Trim vectors
    vocab = load_vocab(filename_words)
    for dim_word in dim_words:
        export_trimmed_word_vectors(vocab, filename_word_vec.format(dim_word),
                                     filename_word_vec_trimmed.format(dim_word), dim_word)

class CoNLLDataset(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """
    def __init__(self, filename, processing_word=None, processing_tag=None,
                 max_iter=None, which_tags=-1):
        """
        :param filename: path to the file
        :param processing_word: (optional) function that takes a word as input
        :param processing_tag: (optional) function that takes a tag as input
        :param max_iter: (optional) max number of sentences to yield
        :param which_tags: (optional) which tagging scheme to use (-1 -2 -3 or 3 2 1 for task 1 2 3 respectively)
        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.which_tags = which_tags
        self.length = None


    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split()
                    word, tag = ls[0],ls[self.which_tags]
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]


    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def coNLLDataset_full(filename, processing_word=None, processing_tag=None, max_iter=None, which_tags=-1):
    """
    Same as above but simply processes all datasets and returns full lists of X and y in memory (no yield).

    :param filename: path to the file
    :param processing_word: (optional) function that takes a word as input
    :param processing_tag: (optional) function that takes a tag as input
    :param max_iter: (optional) max number of sentences to yield
    :param which_tags: (optional) which tagging scheme to use (-1 -2 -3 or 3 2 1 for task 1 2 3 respectively)
    :return X,y: lists of words and tags in sequences
    """


    X,y = [], []

    niter = 0
    with open(filename) as f:
        words, tags = [], []
        for line in f:
            line = line.strip()
            if (len(line) == 0 or line.startswith("-DOCSTART-")):
                if len(words) != 0:
                    niter += 1
                    if max_iter is not None and niter > max_iter:
                        break
                    X.append(words)
                    y.append(tags)
                    words, tags = [], []
            else:
                ls = line.split()
                word, tag = ls[0],ls[which_tags]
                if processing_word is not None:
                    word = processing_word(word)
                if processing_tag is not None:
                    tag = processing_tag(tag)
                words += [word]
                tags += [tag]

    return X,y


def get_vocabs(datasets):
    """
    Build vocabulary from an iterable of datasets objects

    :param datasets: datasets: a list of dataset objects
    :return: a set of all the words in the dataset
    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags


def get_char_vocab(dataset):
    """
    Build char vocabulary from an iterable of datasets objects

    :param dataset: dataset: a iterator yielding tuples (sentence, tags)
    :return: a set of all the characters in the dataset
    """
    vocab_char = set()
    for words, _ in dataset:
        for word in words:
            vocab_char.update(word)

    return vocab_char


def get_vec_vocab(filename):
    """
    Load vocab from file

    :param filename: filename: path to the word vectors
    :return: vocab: set() of strings
    """
    print("Building vocab...")
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def write_vocab(vocab, filename):
    """
    Writes a vocab to a file, one word per line.

    :param vocab: iterable that yields word
    :param filename: path to vocab file
    :return: None (write a word per line)
    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """
    Loads vocab from a file

    :param filename: (string) the format of the file must be one word per line
    :return: dict[word] = index
    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise MyIOError(filename)
    return d


def export_trimmed_word_vectors(vocab, word_filename, trimmed_filename, dim):
    """
    Saves word vectors in numpy array

    :param vocab: dictionary vocab[word] = index
    :param word_filename: a path to a word file
    :param trimmed_filename: a path where to store a matrix in npy
    :param dim: (int) dimension of embeddings
    :return: None
    """
    embeddings = np.zeros([len(vocab), dim])
    with open(word_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_word_vectors(filename):
    """
    Get word vectors

    :param filename: path to the npz file
    :return: matrix of embeddings (np array)
    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)


def get_processing_word(vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False, allow_unk=True):
    """
    Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.
    Note that only known chars from train are used (i.e. chars for which we have learned an embedding, and only known words
    are used. Unknown words are featured with the UNK word vector. Note that this solution prevents learning new embeddings for them,
    because either a word was seen at training, or it is impossible do deal with properly..).

    :param vocab_words: dict[word] = idx
    :param vocab_chars: dict[char] = idx
    :param lowercase: if to transform to lowercase
    :param chars: if to export characters too
    :param allow_unk: if to allow for the use of the UNK token
    :return: f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)
    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Pads to the right, at the end of the sequence.

    :param sequences: a generator of list or tuple
    :param pad_tok: the char to pad with
    :param max_length: the maximum length of a sequence
    :return: a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Pads to the right, at the end of the sequence, at levels 1 (just words) and 2 (both words and characters)

    :param sequences: a generator of list or tuple
    :param pad_tok: the char to pad with
    :param nlevels: "depth" of padding, for the case where we have characters ids
    :return: a list of list where each sublist has same length
    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    Yields data in minimatches.

    :param data: generator of (sentence, tags) tuples
    :param minibatch_size: (int)
    :return: list of tuples
    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


def get_chunk_type(tok, idx_to_tag):
    """
    Return chunk type

    :param tok: id of token, ex 4
    :param idx_to_tag: dictionary {4: "B-PER", ...}
    :return: tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """
    Given a sequence of tags, group entities and their position

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]


    :param seq: [4, 4, 0, 0, ...] sequence of labels
    :param tags: dict["O"] = 4
    :return: list of (chunk_type, chunk_start, chunk_end)
    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "b":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def conv1d(input_, output_size, width=3, stride=1):
    """
    1d convolution for texts, from: https://medium.com/@TalPerry/convolutional-methods-for-text-d5260fd5675f

    :param input_: A tensor of embedded tokens with shape [batch_size,max_length,embedding_size]
    :param output_size: The number of feature maps we'd like to calculate
    :param width: The filter width
    :param stride: The stride
    :return: A tensor of the convolved input with shape [batch_size,max_length,output_size]
    """
    inputSize = input_.get_shape()[-1] # How many channels on the input (The size of our embedding for instance)

    # This is where we make our text an image of height 1
    input_ = tf.expand_dims(input_, axis=1) # Change the shape to [batch_size,1,max_length,embedding_size]

    # Make sure the height of the filter is 1
    filter_ = tf.get_variable("conv_filter_%d_%d" % (width,stride), shape=[1, width, inputSize, output_size])

    # Run the convolution as if this were an image
    convolved = tf.nn.conv2d(input_, filter=filter_, strides=[1, 1, stride, 1], padding="SAME")

    # Remove the extra dimension, i.e. make the shape [batch_size,max_length,output_size]
    result = tf.squeeze(convolved, axis=1)
    return result