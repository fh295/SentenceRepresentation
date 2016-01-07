import numpy
import cPickle as pkl

import gzip

from nltk.tokenize import wordpunct_tokenize
import pdb


class TextIterator:
    def __init__(self, source, 
                 source_dict, 
                 batch_size=128, 
                 maxlen=100,
                 n_words_source=-1):
        if source.endswith('gz'):
            self.source = gzip.open(source, 'r')
        else:
            self.source = open(source, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        ii = 0

        try:
            for ii in xrange(self.batch_size):
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                ss = wordpunct_tokenize(ss.decode('utf-8').strip())
                ss = [self.source_dict[w] if w in self.source_dict else 1 for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]

                if self.maxlen > 0 and len(ss) > self.maxlen:
                    continue

                source.append((numpy.array(ss)))
        except IOError:
            self.end_of_data = True

        if len(source) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source


class TextIteratorMSRP:
    def __init__(self, source, 
                 source_dict, 
                 batch_size=128, 
                 maxlen=100,
                 n_words_source=-1):
        if source.endswith('gz'):
            self.source = gzip.open(source, 'r')
        else:
            self.source = open(source, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        ii = 0

        try:
            for ii in xrange(self.batch_size//2):
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                (ss1,ss2) = tuple(ss.split('\t')[3:])
                ss1,ss2 = wordpunct_tokenize(ss1.lower().decode('utf-8').strip()),\
                           wordpunct_tokenize(ss2.lower().decode('utf-8').strip())
                ss1 = [self.source_dict[w] if w in self.source_dict else 1 for w in ss1]
                ss2 = [self.source_dict[w] if w in self.source_dict else 1 for w in ss2]
                if self.n_words_source > 0:
                    ss1 = [w if w < self.n_words_source else 1 for w in ss1]
                    ss2 = [w if w < self.n_words_source else 1 for w in ss2]
                if self.maxlen > 0 and len(ss1) > self.maxlen and len(ss2) > maxlen:
                    continue

                source.append((numpy.array(ss1)))
                source.append((numpy.array(ss2)))
        except IOError:
            self.end_of_data = True

        if len(source) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source


class MSRPTextIterator:
    def __init__(self, source, 
                 source_dict, 
                 batch_size=64, 
                 maxlen=100,
                 n_words_source=-1):
        if source.endswith('gz'):
            self.source = gzip.open(source, 'r')
        else:
            self.source = open(source, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        ii = 0

        try:
            for ii in xrange(self.batch_size):
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                (ss1,ss2) = tuple(ss.split('\t')[3:])
                ss1,ss2 = wordpunct_tokenize(ss1.lower().decode('utf-8').strip()),\
                           wordpunct_tokenize(ss2.lower().decode('utf-8').strip())
                ss1 = [self.source_dict[w] if w in self.source_dict else 1 for w in ss1]
                ss2 = [self.source_dict[w] if w in self.source_dict else 1 for w in ss2]
                if self.n_words_source > 0:
                    ss1 = [w if w < self.n_words_source else 1 for w in ss1]
                    ss2 = [w if w < self.n_words_source else 1 for w in ss2]
                if self.maxlen > 0 and len(ss1) > self.maxlen and len(ss2) > maxlen:
                    continue

                source.append((numpy.array(ss1), numpy.array(ss2)))
        except IOError:
            self.end_of_data = True

        if len(source) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source









