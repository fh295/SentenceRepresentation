import cPickle as pkl
import gzip
import os
import sys
import time

import threading
import Queue
import logging

import numpy

import theano
import theano.tensor as T

from text_iterator import TextIterator

logger = logging.getLogger(__name__)

def load_data(valid_path=None, test_path=None, batch_size=128):
    ''' 
    Loads the dataset
    '''
    path='../Files/books_test_small.txt'
    dict_path='../Files/books_test_small.txt.dict.pkl'

    #############
    # LOAD DATA #
    #############

    print '... initializing data iterators'

    train = TextIterator(path, dict_path, batch_size=batch_size, maxlen=-1, n_words_source=-1)
    valid = TextIterator(valid_path, dict_path, batch_size=batch_size, maxlen=-1, n_words_source=-1) if valid_path else None
    test = TextIterator(test_path, dict_path, batch_size=batch_size, maxlen=-1, n_words_source=-1) if test_path else None

    return train, valid, test


