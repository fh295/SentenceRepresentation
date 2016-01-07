import argparse
import numpy
import cPickle 
import operator
from collections import defaultdict
from nltk.tokenize import wordpunct_tokenize


def add_line(textline, D):
    words = wordpunct_tokenize(textline.lower().strip())
    for w in words:
        if w.isalpha():
            D[w] += 1
    return D
    
def get_fdist(filename):
    D = defaultdict(int)
    data = open(filename)
    for ss in data:
        D = add_line(ss,D)
    return D

def build_dictionary(path):
    D = {}
    FD = get_fdist(path)
    oFD = sorted_x = sorted(FD.items(), key=operator.itemgetter(1), reverse=True)
    for ii,(w,f) in enumerate(oFD):
        D[w] = ii
    return D

def save_dictionary(outpath,D):
    with open(outpath,'w') as out:
        cPickle.dump(D,out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default=None)
    args = parser.parse_args()

    D = build_dictionary(args.filepath)
    save_dictionary(args.filepath+'.dict.pkl',D)

