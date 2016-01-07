git clone https://github.com/piskvorky/gensim.git

cd gensim
git checkout 485d024e119d0ff818ea6651042748a3bff21081
cd ..

export PYTHONPATH="${PYTHONPATH}:$PWD/gensim"

cp word2vec_inner.pyx word2vec.py gensim/gensim/models/ 

cd gensim/gensim/models
cython -a word2vec_inner.pyx 
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
-I/usr/include/python2.7 -o word2vec_inner.so word2vec_inner.c
cd ../../../

python trainFastSent.py --corpus ../../Fastsent/data/books_large_70m.txt  --dim 300


# possible options
# --dim: the dimension of the word (and sentence) embeddings
# --corpus: path/to/corpus/ (see README for corpus specifications)
# --min_count the smallest possible word frequency embedded by the model. Words with frequency lower than this number will be ignored. Default = 10. For smaller training corpora, make this number smaller. 
# --sample: downsampling of frequent words. See gensim word2vec documentation for details. Default is no downsampling. 
# --autoencode: do you autoencode the current sentence or not. See the paper for more details. 
# --savedir: path to a directory where you want to save the model. The model filename is generated automatically from the model options. 

# You can make further changes to the model by changing the options in the gensim.word2vec.Word2Vec class. Just change 
# line 45 of trainFastSent.py
