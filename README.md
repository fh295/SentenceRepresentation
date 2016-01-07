# SentenceRepresentation

This code acompanies the paper 'Learning Sentence Representations from Unlabelled Data' Felix Hill, KyungHyun Cho and Anna Korhonen 2016. 

To train a FastSent model
=========================

Move to the FastSent directory. The code is based on a small change to the gensim code. You can find out more about gensim at https://radimrehurek.com/gensim/, it is really good, and the contributors to gensim deserve 99% of the credit for this implementation. 

To train a fastsent model, just run ./train_model.sh. The script checks out a particular version of gensim (the version on top of which we made the change) and copies in our modifications. 

For things to work you will need to check the following: 

- You might need to run 'chmod +x train_model.sh' in the terminal to give permission to run the executable file. 

- You will need to change the command-line options in train_model.sh. Most importantly, the option --corpus must be followed by a path to your corpus file


The corpus
---------
 This must be a plaintext file with each sentence on a new line. It's no problem to have full-stops at the end of each sentence. Implement any pre-processing on this file before training. We used the Toronto-Books Corpus (http://www.cs.toronto.edu/~mbweb/) as is (the only pre-processing was lower-casing). 


To train an SDAE
===============

Move to the SDAE directory. 

- Build a dictionary for your corpus. Run 'python build_dictionary.py --filename path/to/your/corpus.txt. The corpus should meet the same specifications as for FastSent (above). 

- Open book.py and set the path to the dictionary to be the new dictionary created by this process (you only need to do this once). Set the path to the corpus to point to the corpus itself. 

- Open train_book.py. Here you should set the hyperparams for the model you want to train. Comments in the file indicate what they do. 


Pre-trained embeddings
---------------------
To train a model with pre-trained word embeddings (which are mapped into the RNN via a learned mapping but not updated during training) you need to put your word embeddings in the following form. 

{'theword': numpy.array([a word embedding],dtype=float32) for 'theword' in your vocabulary}

This object (a python dictionary) needs to be saved as a pickle file (using cPickle). Then set use_preemb to True 





 

