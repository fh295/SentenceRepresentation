import numpy

from desent import train

def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    trainerr, validerr, testerr = train(saveto=params['model'][0],
                                        reload_=params['reload'][0],
                                        corruption=params['corruption'][0],
                                        corruption_prob=params['corruption_prob'][0],
                                        dim_word=params['dim_word'][0],
                                        dim=params['dim'][0],
                                        n_words=params['n-words'][0],
                                        decay_c=params['decay-c'][0],
                                        param_noise=params['param-noise'][0],
                                        lrate=params['learning-rate'][0],
                                        optimizer=params['optimizer'][0], 
                                        maxlen=100,
                                        batch_size=16,
                                        dictionary = params['dictionary'][0],
                                        valid_batch_size=16,
                                        validFreq=1000,
                                        dispFreq=1,
                                        saveFreq=1000,
                                        clip_c=params['clip-c'][0],
                                        encoder=params['encoder'][0],
                                        use_preemb=params['use_preemb'][0],
                                        dataset='book', 
                                        use_dropout=params['use-dropout'][0],
                                        embeddings=params['embeddings'][0])
    return validerr

if __name__ == '__main__':
    main(0, {'model': ['your_name_for_saved_model.npz'],
             'dim_word': [100],
             'dim': [2400],
             'n-words': [30000], 
             'optimizer': ['adam'],
             'decay-c': [0.], 
             'clip-c': [2.],
             'param-noise': [0.], 
             'corruption': [['_mask','_shuffle']],
             'corruption_prob': [[0.1,0.1]],
             'use-dropout': [False],
             'learning-rate': [0.01],
             'encoder': ['gru'],
             'decoder': ['gru_cond'],
             'use_preemb': [True],
             'reload': [False],
             'dictionary': ['../Files/books_test_small.txt.dict.pkl'],
             'embeddings': ['../Files/D_medium_cbow_pdw_8B.pkl'],
             'valid_text': ['../Files/newsdev.tok.en'],
             'test_text': ['../Files/newstest.tok.en']})

# model: path/name.npz for your trained model
# dim_word: dimension of word embedding (if use_preemb = False)
# dim: dimension of RNN (GRU) hidden states (and sentence vector)
# n-words: size of vocab
# optimizer: optimization algorithm
# decay-c: GRU gates decay
# clip-c: gradient clipping
# corrpution: noise process  _mask => omit words _shuffle => shuffle (p_o and p_x in paper)
# encoder/decoder: architecture for encoder and decoder
# use_preemb: set to True to use pre-trained word embeddings
# reload: set to True if reloading a model (in the model: option) for more training
# embeddings: path to pre-trained word embedddings (if use_preemb = True)
# valid/test_text: text (sentences) to compute validation error during training

