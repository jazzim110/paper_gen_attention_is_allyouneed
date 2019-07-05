import time
import numpy as np
import tensorflow as tf

from config import *

class Vocab:
    def __init__(self, token2idx=None, idx2token=None):
        self.token2idx = token2idx or dict()
        self.idx2token = idx2token or dict()

    def new_token(self, token):
        # token : word of dataset
        # return : index of token
        if token not in self.token2idx:
            index = len(self.token2idx)
            self.token2idx[token] = index
            self.idx2token[index] = token
        return self.token2idx[token]

    def get_token(self, index):
        # index : position number of token
        # return : word or character of index
        return self.idx2token[index]

    def get_index(self, token):
        # token : word of dataset
        # return : index of token
        return self.token2idx[token]

def make_data(file_name=None):
    # --------------------------- Input --------------------------- #
    # file_name : file name of preprocessed data of TED video script

    # --------------------------- Output --------------------------- #
    # word_vocab : class, composed of token to index dictionary and reversed dictionary
    # word_list  : total sentences of training data as index list
    # whole_sent : extended list of tatal senteces
    # word_maxlen: max length of vocabulary dictionary words

    word_vocab = Vocab()

    word_list = list()
    whole_sent = list()

    EOS = '|' # End of sentence token
    UNK = '<UNK>' # For unknown word at dev and test set

    word_vocab.new_token(EOS) # End of sentence token to index 0
    word_vocab.new_token(UNK) # Unknown word will appear at dev and test set

    word_maxlen = 0

    start = time.time()
    with open(FLAGS.data_path + file_name, 'r', encoding='utf-8') as f:
        line = f.readlines()

        for one_line in line:
            one_sent = list()

            for word in one_line.split():
                one_sent.append(word_vocab.new_token(word))

                if len(word) > word_maxlen:
                    word_maxlen = len(word) # 61 is the longest word

            # End of Sentence
            one_sent.append(word_vocab.get_index(EOS))

            word_list.append(one_sent)
            whole_sent.extend(one_sent)

    print (file_name + " file making indexing table time: %.3f" % (time.time() - start))
    print ("dictionary size : ", len(word_vocab.token2idx))
    print ("total number of sentences : ", len(word_list))
    print ("max length of the word : ", word_maxlen)
    print ()

    return word_vocab, word_list, whole_sent, word_maxlen

def get_data(en_list, de_list):
    # --------------------------- Input --------------------------- #
    # en_list : 196884 number of sentences at ted video, composed of English language data
    # de_list : 196884 number of sentences at ted video, composed of German language data

    # --------------------------- Output --------------------------- #
    # X : padded results of index list, composed of English lnaguage data (131549, 20)
    # Y : padded results of index list, composed of German language data  (131549, 20)

    source = list()
    target = list()

    for idx in range(len(de_list)):
        # Remove sentence length is longer than 20 words
        if max(len(de_list[idx]), len(en_list[idx])) <= FLAGS.sentence_maxlen:
            source.append(np.array(de_list[idx]))
            target.append(np.array(en_list[idx]))

    # make the shape of Source matrix and Target matrix
    X = np.zeros([len(source), FLAGS.sentence_maxlen], dtype=np.int32)
    Y = np.zeros([len(target), FLAGS.sentence_maxlen], dtype=np.int32)

    # Padding with the shape of (sentence number, sentence length)
    for idx, (x, y) in enumerate(zip(source, target)):
        X[idx, :len(x)] = x
        Y[idx, :len(y)] = y

    print ("Source Matrix Shape (DE):", X.shape)
    print ("Target Matrix Shape (EN):", Y.shape)
    print ()
    print ('------------------------ Show the example case ------------------------')
    print (X[0])
    print (Y[0])

    print ('------------------------ Show the example case ------------------------')
    print (X[10])
    print (Y[10])

    return X, Y

def batch_loader(X, Y):

    reduced_length = len(X) // FLAGS.batch_size * FLAGS.batch_size # 131520

    X = X[:reduced_length]
    Y = Y[:reduced_length]

    print ("Reduced Source Matrix shape (DE):", X.shape)
    print ("Reduced Target Matrix shape (EN):", Y.shape)

    X = np.reshape(X, newshape=(FLAGS.batch_size, -1, FLAGS.sentence_maxlen))
    Y = np.reshape(Y, newshape=(FLAGS.batch_size, -1, FLAGS.sentence_maxlen))

    print ("Shape of Source Matrix (DE):", X.shape)
    print ("Shape of Target Matrix (EN):", Y.shape)

    X = np.transpose(X, axes=(1,0,2))
    Y = np.transpose(Y, axes=(1,0,2))

    print ("Shape of Source Matrix (DE):", X.shape)
    print ("Shape of Target Matrix (EN):", Y.shape)

    # while training, yield the shape of (batch_size, sentence max length) in X and Y
    zip_file = list(zip(X, Y))

    return X, Y, zip_file

def preprocess():
    en_vocab, en_list, en_sent, _ = make_data(file_name='train.en')
    de_vocab, de_list, de_sent, _ = make_data(file_name='train.de')
    X, Y = get_data(en_list, de_list)
    X, Y, zip_file = batch_loader(X, Y)

    return X, Y, en_vocab, de_vocab, zip_file
