"""
Reference parsing model
Borrows from: https://github.com/guillaumegenthial/sequence_tagging
"""

import numpy as np
import tensorflow as tf
import os
from collections import OrderedDict

from sklearn.base import BaseEstimator, ClassifierMixin

from model.data_utils import minibatches, pad_sequences, get_chunks, build_data, \
    export_trimmed_word_vectors, load_vocab, get_processing_word, CoNLLDataset, coNLLDataset_full, conv1d
from model.general_utils import Progbar

class RefModel(BaseEstimator, ClassifierMixin):
    """Model for reference parsing"""

    def __init__(self,processing_word,processing_tag,vocab_chars,vocab_words,vocab_tags,
                 nwords,nchars,ntags,dir_output,dir_model,dim_word=300,dim_char=100,use_pretrained=False,train_embeddings=False,
                 dropout=0.5,batch_size=50,lr_method="adam",lr=0.001,lr_decay=0.9,
                 clip=-1,nepoch_no_imprv=10,l2_reg_lambda=0.0,hidden_size_char=100,hidden_size_lstm=300,
                 use_crf=True,use_chars=True,use_cnn=False,random_state=None):
        """
        Initialize the RefModel by simply storing all the hyperparameters.

        :param processing_word: (function) to process words
        :param processing_tag: (function) to process tags
        :param vocab_chars: (dictionary) of characters
        :param vocab_words: (dictionary) of words
        :param vocab_tags: (dictionary) of tags
        :param nwords: (int) number of words
        :param nchars: (int) number of characters
        :param ntags: (int) number of tags
        :param dir_output: (string) output directory
        :param dir_model: (string) model output directory
        :param dim_word: (int) dimensionality of word embeddings
        :param dim_char: (int) dimensionality of character embeddings
        :param use_pretrained: (bool) if to use pretrained embeddings
        :param train_embeddings: (bool) if to further train embeddings
        :param dropout: (float between 0 and 1) propout percentage
        :param batch_size: (int) batch size
        :param lr_method: (string) learning method (adagrad, sgd, rmsprop)
        :param lr: (float) learning rate
        :param lr_decay: (float between 0 and 1) learning rate
        :param clip: (float) clip rate
        :param nepoch_no_imprv: (int) early stopping number of epochs before interrupting without improvements
        :param l2_reg_lambda: (float) lambda for l2 regularization
        :param hidden_size_char: (int) size of hidden character lstm layer
        :param hidden_size_lstm: (int) size of hidden lstm layer
        :param use_crf: (bool) if to use crf prediction
        :param use_chars: (bool) if to use characters
        :param use_cnn: (bool) if to use cnn over lstm for character embeddings
        :param random_state: (int) random state
        """

        # externals
        self.processing_word = processing_word
        self.processing_tag  = processing_tag
        self.vocab_chars     = vocab_chars
        self.vocab_words     = vocab_words
        self.vocab_tags      = vocab_tags
        self.nwords          = nwords         
        self.nchars          = nchars
        self.ntags           = ntags
        self.dir_output      = dir_output
        self.dir_model       = dir_model

        # embeddings
        self.dim_word = dim_word
        self.dim_char = dim_char
        self.use_pretrained = use_pretrained
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.vocab_tags.items()}

        # training
        self.train_embeddings = train_embeddings
        self._dropout = dropout
        self.batch_size = batch_size
        self.lr_method = lr_method
        self._lr = lr
        self.lr_decay = lr_decay
        self.clip = clip  # if negative, no clipping
        self.nepoch_no_imprv = nepoch_no_imprv
        self.l2_reg_lambda = l2_reg_lambda  # if 0, no l2 regularization

        # model hyperparameters
        self.hidden_size_char = hidden_size_char  # lstm on chars
        self.hidden_size_lstm = hidden_size_lstm  # lstm on word embeddings

        # NOTE: if both chars and crf, only 1.6x slower on GPU
        self.use_crf = use_crf  # if crf, training is 1.7x slower on CPU
        self.use_chars = use_chars  # if char embedding, training is 3.5x slower on CPU
        self.use_cnn = use_cnn  # if to use CNN char embeddings, if not use bi-LSTM

        # embedding files
        self._filename_emb = "../pretrained_vectors/vecs_{}.txt".format(self.dim_word)
        # trimmed embeddings (created with build_data.py)
        self._filename_trimmed = "../pretrained_vectors/vecs_{}.trimmed.npz".format(self.dim_word)
        self.embeddings = (export_trimmed_word_vectors(self._filename_trimmed)
                      if self.use_pretrained else None)

        # extra
        self.random_state = random_state
        self._session = None


    def _add_placeholders(self):
        """Define placeholder entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")

        # l2 regularization
        self.l2_loss = tf.constant(0.0, name="l2_loss")


    def _get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """
        Given some data, pad it and build a feed dictionary

        :param words: (list) of sentences. A sentence is a list of ids of a list of words. A word is a list of ids
        :param labels: (list) of ids
        :param lr: (float) learning rate
        :param dropout: (float) keep prob
        :return: dict {placeholder: value}
        """

        # perform padding of the given data
        if self.use_chars:
            words = [zip(*w) for w in words]
            char_ids,word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths


    def _add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we train the vectors if config train_embeddings is True.
        Otherwise, a random matrix with the correct shape is initialized.

        Note: add a DropoutWrapper to have dropout within cells.
        """

        with tf.variable_scope("words"):
            if self.embeddings is None:
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.nwords, self.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.nchars, self.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                # now becomes batch size * max sentence length, char in word, dim_char
                char_embeddings = tf.reshape(char_embeddings,
                                             shape=[s[0] * s[1], s[-2], self.dim_char])

                if self.use_cnn:
                    widths = [2,3,5]
                    strides = [1]
                    outputs = list()
                    for w in widths:
                        for st in strides:
                            with tf.name_scope("conv-maxpool-%d-%d" % (w,st)):
                                output = conv1d(char_embeddings, self.hidden_size_char, width=w, stride=st)
                                output = tf.reduce_max(tf.nn.relu(output), 1)  # activation and max pooling to have 1 feature vector per word
                                outputs.append(output)

                    # concat output
                    output = tf.concat(outputs, axis=-1)

                    # shape = (batch size, max sentence length, len(widths)*len(strides) * char hidden size)
                    output = tf.reshape(output,
                                        shape=[s[0], s[1], len(widths)*len(strides) * self.hidden_size_char])
                    output = tf.nn.dropout(output, self.dropout)

                else:
                    # bi-LSTM to learn character embeddings
                    # reshape word lengths
                    word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                    # bi lstm on chars
                    cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size_char,
                            state_is_tuple=True)
                    cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size_char,
                            state_is_tuple=True)
                    _output = tf.nn.bidirectional_dynamic_rnn(
                            cell_fw, cell_bw, char_embeddings,
                            sequence_length=word_lengths, dtype=tf.float32)

                    # read and concat output
                    _, ((_, output_fw), (_, output_bw)) = _output
                    output = tf.concat([output_fw, output_bw], axis=-1)

                    # shape = (batch size, max sentence length, 2*char hidden size)
                    output = tf.reshape(output,
                            shape=[s[0], s[1], 2*self.hidden_size_char])
                    output = tf.nn.dropout(output, self.dropout)

                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)


    def _add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.

        Note: add a DropoutWrapper to have dropout within cells.
        """

        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        # act here to expand to multiple outputs and to add attention
        with tf.variable_scope("pred"):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.hidden_size_lstm, self.ntags])

            b = tf.get_variable("b", shape=[self.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())
            # l2 regularization
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.ntags])


    def _add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With CRF, as the inference is coded
        in python and not in pure tensorflow, we have to make the prediction
        outside the graph.

        Note: this is no longer the case, see https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/crf.
        """

        if not self.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)


    def _add_loss_op(self):
        """Defines the loss"""

        if self.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood) + self.l2_reg_lambda * self.l2_loss
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss


    def _add_train_op(self, lr_method, lr, loss, clip=-1):
        """
        Defines self.train_op that performs an update on a batch

        :param lr_method: (string) sgd method, for example "adam"
        :param lr: (tf.placeholder) tf.float32, learning rate
        :param loss: (tensor) tf.float32 loss to minimize
        :param clip: (python float) clipping of gradient. If < 0, no clipping
        :return: None
        """

        _lr_m = lr_method.lower() # lower to make sure

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam': # sgd method
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0: # gradient clipping if clip is positive
                grads, vs     = zip(*optimizer.compute_gradients(loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)


    def _predict_batch(self, words):
        """
        Predict for a batch of data

        :param words: (list) of sentences
        :return: (list) of labels for each sentence
            sequence_length
        """

        fd, sequence_lengths = self._get_feed_dict(words, dropout=1.0)

        if self.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self._session.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in viterbi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self._session.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths


    def _run_epoch(self, X_train, y_train, X_dev, y_dev, epoch):
        """
        Performs one complete pass over the train set and evaluate on dev

        :param X_train: (list) with training data
        :param y_train: (list) with training labels
        :param X_dev: (list) with testing data
        :param y_dev: (list) with testing labels
        :param epoch: (int) which epoch it is
        :return: (python float) score to select model on, higher is better
        """

        # progbar stuff for logging
        batch_size = self.batch_size
        nbatches = (len(X_train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        rnd_idx = np.random.permutation(len(X_train))
        for i, rnd_indices in enumerate(np.array_split(rnd_idx, len(X_train) // batch_size)):
            words, labels = [X_train[x] for x in list(rnd_indices)], [y_train[y] for y in list(rnd_indices)]
            fd, _ = self._get_feed_dict(words, labels, self._lr, self._dropout)

            _, train_loss = self._session.run(
                    [self.train_op, self.loss], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                # loss
                loss_summary = self._loss_summary.eval(feed_dict=fd)
                self._file_writer.add_summary(loss_summary, epoch * nbatches + i)
                # train eval
                metrics = self._run_evaluate(words, labels)
                summary = tf.Summary()
                summary.value.add(tag='precision_train', simple_value=metrics["p"])
                summary.value.add(tag='recall_train', simple_value=metrics["r"])
                summary.value.add(tag='f1_train', simple_value=metrics["f1"])
                summary.value.add(tag='accuracy_train', simple_value=metrics["acc"])
                self._file_writer.add_summary(summary, epoch * nbatches + i)
                # test eval
                metrics = self._run_evaluate(X_dev, y_dev)
                summary = tf.Summary()
                summary.value.add(tag='precision_test', simple_value=metrics["p"])
                summary.value.add(tag='recall_test', simple_value=metrics["r"])
                summary.value.add(tag='f1_test', simple_value=metrics["f1"])
                summary.value.add(tag='accuracy_test', simple_value=metrics["acc"])
                self._file_writer.add_summary(summary, epoch)

        # final epoch test eval
        metrics = self._run_evaluate(X_dev, y_dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        print(msg)

        return metrics["f1"]


    def _run_evaluate(self, X_dev, y_dev):
        """
        Evaluates performance on test set

        :param X_dev:(list) with dev data
        :param y_dev: (list) with dev labels
        :return: (dict) metrics["acc"] = 98.4, ...
        """

        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.

        rnd_idx = np.random.permutation(len(X_dev))
        for rnd_indices in np.array_split(rnd_idx, len(X_dev) // self.batch_size):
            words, labels = [X_dev[x] for x in list(rnd_indices)], [y_dev[y] for y in list(rnd_indices)]
            labels_pred, sequence_lengths = self._predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return OrderedDict({"acc": 100*acc, "f1": 100*f1, "p": p, "r": r})


    def _reinitialize_weights(self, scope_name):
        """Reinitializes the weights of a given layer

        :param scope_name: (string) scope of variables to reinitialize
        """

        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self._session.run(init)


    def _initialize(self):
        """Initialize the variables"""

        print("Initializing tf session")
        self._init = tf.global_variables_initializer()
        self._saver = tf.train.Saver()


    def restore_session(self):
        """Reload weights into session"""
        self._graph = tf.Graph()
        with self._graph.as_default():
            self.build()
        self._session = tf.Session(graph=self._graph)
        self._saver.restore(self._session, self.dir_model)


    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.dir_model):
            os.makedirs(self.dir_model,exist_ok=True)
        self._saver.save(self._session, self.dir_model)


    def add_summary(self):
        """Defines variables for Tensorboard"""
        self._loss_summary = tf.summary.scalar('loss', self.loss)
        self._file_writer = tf.summary.FileWriter(self.dir_output,
                self._session.graph)


    def close_session(self):
        """Closes the session"""
        if self._session:
            self._session.close()


    def _get_model_params(self):
        """From: https://github.com/ageron/handson-ml/blob/master/11_deep_learning.ipynb
        Get all variable values (used for early stopping, faster than saving to disk)"""

        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}


    def _restore_model_params(self, model_params):
        """From: https://github.com/ageron/handson-ml/blob/master/11_deep_learning.ipynb
        Set all variables to the given values (for early stopping, faster than loading from disk)

        :param model_params: (dict) parameters of the model to restore
        """

        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign")
                      for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        fd = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=fd)


    def build(self):
        """Builds the computational graph"""

        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        # specific functions
        self._add_placeholders()
        self._add_word_embeddings_op()
        self._add_logits_op()
        self._add_pred_op()
        self._add_loss_op()

        # generic functions that add training op and initialize vars
        self._add_train_op(self.lr_method, self.lr, self.loss, self.clip)
        self._initialize() # initialize vars and saver, session is still not there


    def fit(self, X, y, X_valid=None, y_valid=None, nepochs=100):
        """
        Performs training with early stopping and lr exponential decay

        :param X: (list) data
        :param y: (list) labels
        :param X_valid: (list) validation data
        :param y_valid: (list) validation data
        :param nepochs: (int) number of epochs to run for
        :return: self (model, instance of RefModel)
        """

        self.close_session()
        self._graph = tf.Graph()
        with self._graph.as_default():
            self.build()

        self.best_score = 0
        nepoch_no_imprv = 5 # for early stopping, this should be passed as a parameter
        best_params = None

        self._session = tf.Session(graph=self._graph)
        with self._session.as_default():
            self._init.run()
            self.add_summary()  # tensorboard
            for epoch in range(nepochs):
                print("Epoch {:} out of {:}".format(epoch + 1, nepochs))

                score = self._run_epoch(X, y, X_valid, y_valid, epoch)
                self._lr *= self.lr_decay # decay learning rate

                # early stopping and saving best parameters
                if score >= self.best_score:
                    best_params = self._get_model_params()
                    nepoch_no_imprv = 0
                    self.best_score = score
                    print("- new best score!")
                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.nepoch_no_imprv:
                        print("- early stopping {} epochs without "\
                                "improvement".format(nepoch_no_imprv))
                        break

            # If we used early stopping then rollback to the best model found
            if best_params:
                self._restore_model_params(best_params)
            return self


    def predict(self, words_raw):
        """
        Returns list of predicted tags

        :param words_raw: (list) of words (string), just one sentence (no batch)
        :return preds: (list) of tags (string), one for each word in the sentence
        """

        words = [self.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self._predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds


    def evaluate(self, X_dev, y_dev):
        """
        Evaluate model on test set

        :param X_dev: (list) dev data
        :param y_dev: (list) dev labels
        :return: (dict) of metrics
        """

        metrics = self._run_evaluate(X_dev, y_dev)
        return metrics


if __name__ == "__main__":

    # Example of usage

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

    # vocabs (created with build_data.py)
    filename_words = "working_dir/words.txt"
    filename_words_ext = "working_dir/words_ext.txt"
    filename_tags = "working_dir/tags.txt"
    filename_chars = "working_dir/chars.txt"

    # build data (just to test model)
    build_data(filename_dev, filename_test, filename_train, [300], filename_words,
               filename_words_ext, filename_tags, filename_chars,
               filename_word="../pretrained_vectors/vecs_{}.txt",
               filename_word_vec_trimmed="../pretrained_vectors/vecs_{}.trimmed.npz",
               which_tags=which_tags)

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
    X_dev, y_dev = coNLLDataset_full(filename_dev, processing_word, processing_tag, max_iter, which_tags)
    X_train, y_train = coNLLDataset_full(filename_train, processing_word, processing_tag, max_iter, which_tags)
    X_valid, y_valid = coNLLDataset_full(filename_test, processing_word, processing_tag, max_iter, which_tags)

    print("Size of train, test and valid sets (in number of sentences): ")
    print(len(X_train), " ", len(y_train), " ", len(X_dev), " ", len(y_dev), " ", len(X_valid), " ", len(y_valid))

    model = RefModel(processing_word=processing_word,processing_tag=processing_tag,vocab_chars=vocab_chars,
                     vocab_words=vocab_words,vocab_tags=vocab_tags,nwords=nwords,nchars=nchars,
                     ntags=ntags,dir_output=dir_output,dir_model=dir_model,use_chars=use_chars,random_state=0,
                     use_pretrained=True, hidden_size_char=50, batch_size=100, lr_decay=1, l2_reg_lambda=0,
                     use_crf=True, use_cnn=False, dim_word=300, hidden_size_lstm=200, lr=0.001,
                     train_embeddings=True, dim_char=100, lr_method="rmsprop")

    fitted = model.fit(X_train, y_train, X_dev, y_dev, 50)
    print("Final f1 score: ",fitted.best_score)
    print("\nValidation:")
    print(str(fitted.evaluate(X_valid, y_valid)))