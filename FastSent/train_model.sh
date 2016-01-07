git clone https://github.com/piskvorky/gensim.git

cd gensim
git checkout da7f9e30d7a2ceeb9b874948f821cbec7ae439df
cd ..

export PYTHONPATH="${PYTHONPATH}:$PWD/gensim"

cp word2vec_inner.pyx word2vec.py gensim/gensim/models/ 

python trainFastSent.py --corpus ../../Fastsent/data/books_large_70m.txt --dim 300
