'''
Denoising Sentence Autoencoder
Code by KyungHyunCho with some help from Felix Hill
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import cPickle as pkl
import numpy
import copy
import os
import pdb
from scipy import optimize, stats
from collections import OrderedDict
from sklearn.cross_validation import KFold

from nltk.tokenize import wordpunct_tokenize

import book

profile = False
datasets = {'book': book.load_data}
             
def get_dataset(name):
    return datasets[name]

def prepare_data(seqs_x, seqs_xn, maxlen=None, n_words=30000):
    # x: a list of sentences
    try:
        lengths_x = [len(s) for s in seqs_x]
        lengths_xn = [len(s) for s in seqs_xn]
    except:
        pdb.set_trace()

    if maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        new_seqs_xn = []
        new_lengths_xn = []
        for l_x, s_x, l_xn, s_xn in zip(lengths_x, seqs_x, lengths_xn, seqs_xn):
            if l_x < maxlen and l_xn < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_xn.append(s_xn)
                new_lengths_xn.append(l_xn)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_xn = new_lengths_xn
        seqs_xn = new_seqs_xn

        if len(lengths_x) < 1 or len(lengths_xn) < 1:
            return None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_xn = numpy.max(lengths_xn) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    xn = numpy.zeros((maxlen_xn, n_samples)).astype('int64')
    xn_mask = numpy.zeros((maxlen_xn, n_samples)).astype('float32')

    for idx, s_x in enumerate(seqs_x):
        s_x[numpy.where(s_x >= n_words-1)] = 1
        x[:lengths_x[idx],idx] = s_x
        x_mask[:lengths_x[idx]+1,idx] = 1.

    for idx, s_x in enumerate(seqs_xn):
        s_x[numpy.where(s_x >= n_words-1)] = 1
        xn[:lengths_xn[idx],idx] = s_x
        xn_mask[:lengths_xn[idx]+1,idx] = 1.

    return x, x_mask, xn, xn_mask

# corruption process
def _mask(x, degree=0.1, use_preemb=False):
    n_words = x.shape[0]
    rndidx = numpy.random.permutation(n_words)
    n_corr = numpy.round(numpy.float32(n_words) * degree)

    corridx = rndidx[:n_corr]

    x_noise = copy.copy(x)
    for ci in corridx:
        if use_preemb:
            x_noise[ci] = x_noise[ci] * 0.
        else:
            x_noise[ci] = 1
        
    return x_noise

def _shuffle(x, degree=0.1, use_preemb=False):
    n_words = x.shape[0]
    rndidx = numpy.random.permutation(n_words-1)
    n_corr = numpy.round(numpy.float32(n_words) * degree)

    corridx = rndidx[:n_corr]

    x_noise = copy.copy(x)
    for ci in corridx:
        oo = copy.copy(x_noise[ci+1])
        x_noise[ci+1] = x_noise[ci]
        x_noise[ci] = oo
        
    return x_noise

def _remove(x, degree=0.1, use_preemb=False):
    n_words = x.shape[0]
    rndidx = numpy.random.permutation(n_words-1)
    n_corr = numpy.round(numpy.float32(n_words) * degree)

    corridx = set(rndidx[:n_corr])

    x_noise = []
    for ii, xx in enumerate(x):
        if ii not in corridx:
            x_noise.append(xx)
        
    return x_noise


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise, 
            state_before * trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype),
            state_before * 0.5)
    return proj

# make prefix-appended name
def _p(pp, name):
    return '%s_%s'%(pp, name)

# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive'%kk)
        params[kk] = pp[kk]

    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'), 
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          }

def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin,nout=None, scale=0.01, orth=True):
    if nout == None:
        nout = nin
    if nout == nin and orth:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')

def tanh(x):
    return tensor.tanh(x)

def rectifier(x):
    return tensor.maximum(0., x)

def linear(x):
    return x

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None, orth=True):
    if nin == None:
        nin = options['dim_proj']
    if nout == None:
        nout = options['dim_proj']
    params[_p(prefix,'W')] = norm_weight(nin, nout, scale=0.01, orth=orth)
    params[_p(prefix,'b')] = numpy.zeros((nout,)).astype('float32')

    return params

def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix,'W')])+tparams[_p(prefix,'b')])

# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None, hiero=False):
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']
    if not hiero:
        W = numpy.concatenate([norm_weight(nin,dim),
                               norm_weight(nin,dim)], axis=1)
        params[_p(prefix,'W')] = W
        params[_p(prefix,'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U

    Wx = norm_weight(nin, dim)
    params[_p(prefix,'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[_p(prefix,'Ux')] = Ux
    params[_p(prefix,'bx')] = numpy.zeros((dim,)).astype('float32')

    return params

def gru_layer(tparams, state_below, options, prefix='gru', mask=None, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix,'Ux')].shape[1]

    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    U = tparams[_p(prefix, 'U')]
    Ux = tparams[_p(prefix, 'Ux')]

    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h#, r, u, preact, preactx

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    rval, updates = theano.scan(_step, 
                                sequences=seqs,
                                outputs_info = [tensor.alloc(0., n_samples, dim)],
                                                #None, None, None, None],
                                non_sequences = [tparams[_p(prefix, 'U')], 
                                                 tparams[_p(prefix, 'Ux')]],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval

# Conditional GRU layer without Attention
def param_init_gru_cond(options, params, prefix='gru_cond', nin=None, dim=None, dimctx=None):
    if nin == None:
        nin = options['dim']
    if dim == None:
        dim = options['dim']
    if dimctx == None:
        dimctx = options['dim']

    params = param_init_gru(options, params, prefix, nin=nin, dim=dim)

    # context to LSTM
    Wc = norm_weight(dimctx,dim*2)
    params[_p(prefix,'Wc')] = Wc

    Wcx = norm_weight(dimctx,dim)
    params[_p(prefix,'Wcx')] = Wcx

    return params

def gru_cond_layer(tparams, state_below, options, prefix='gru', 
                   mask=None, context=None, one_step=False, 
                   init_memory=None, init_state=None, 
                   context_mask=None,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    # initial/previous state
    if init_state == None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context 
    assert context.ndim == 2, 'Context must be 2-d: #sample x dim'
    pctx_ = tensor.dot(context, tparams[_p(prefix,'Wc')])
    pctxx_ = tensor.dot(context, tparams[_p(prefix,'Wcx')])

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    def _step_slice(m_, x_, xx_, h_, pctx_, pctxx_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_
        preact += pctx_
        preact = tensor.nnet.sigmoid(preact)

        r = _slice(preact, 0, dim)
        u = _slice(preact, 1, dim)

        preactx = tensor.dot(h_, Ux)
        preactx *= r
        preactx += xx_
        preactx += pctxx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]] 

    if one_step:
        rval = _step(*(seqs+[init_state, pctx_, pctxx_]+shared_vars))
    else:
        rval, updates = theano.scan(_step, 
                                    sequences=seqs,
                                    outputs_info=[init_state], 
                                    non_sequences=[pctx_,
                                                   pctxx_]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval

# initialize all parameters
def init_params(options):
    params = OrderedDict()
    # embedding: shared
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])
    # encoder
    params = get_layer(options['encoder'])[0](options, params, prefix='encoder', 
                                              nin=options['dim_word'], 
                                              dim=options['dim'])
    init_state = get_layer('ff')[0](options, params, prefix='ff_state',
                                    nin=options['dim'], nout=options['dim'])
    # decoder
    params = get_layer(options['decoder'])[0](options, params, prefix='decoder', 
                                              nin=options['dim_word'], 
                                              dim=options['dim'],
                                              dimctx=options['dim'])
    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit', 
                                nin=options['dim'], nout=options['n_words'])

    return params

# build a training model
def build_model(tparams, options):
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # clean string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    # noisy string: #words x #samples
    if options['use_preemb']:
        x_noise = tensor.tensor3('x_noise', dtype='float32')
    else:
        x_noise = tensor.matrix('x_noise', dtype='int64')
    xn_mask = tensor.matrix('x_noise_mask', dtype='float32')

    n_timesteps = x_noise.shape[0]
    n_samples = x_noise.shape[1]

    # word embedding (source)
    if options['use_preemb']:
        emb = x_noise
    else:
        emb = tparams['Wemb'][x_noise.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
    # encoder
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            mask=xn_mask)
    ctx = proj[0][-1] # the last hidden state is the context
    init_state = get_layer('ff')[1](tparams, ctx, options, 
                                    prefix='ff_state', activ='tanh')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (target)
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted
    # decoder
    proj = get_layer(options['decoder'])[1](tparams, emb, options, 
                                            prefix='decoder', 
                                            mask=x_mask, context=ctx, 
                                            one_step=False, 
                                            init_state=init_state)
    proj_h = proj
    # compute word probabilities
    logit = get_layer('ff')[1](tparams, proj_h, options, prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))
    # cost
    x_flat = x.flatten()
    x_flat_idx = tensor.arange(x_flat.shape[0]) * options['n_words'] + x_flat
    cost = -tensor.log(probs.flatten()[x_flat_idx]+1e-8)
    cost = cost.reshape([x.shape[0],x.shape[1]])
    cost = (cost * x_mask).sum(0)
    cost = cost.mean()

    return trng, use_noise, x, x_mask, x_noise, xn_mask, ctx, cost

# build a sampler
def build_sampler(tparams, options, trng):
    if options['use_preemb']:
        x = tensor.tensor3('x', dtype='float32')
    else:
        x = tensor.matrix('x', dtype='int64')
    xr = x[::-1]
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (source)
    if options['use_preemb']:
        emb = x
    else:
        emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
    # encoder
    proj = get_layer(options['encoder'])[1](tparams, emb, options, prefix='encoder')

    ctx = proj[0][-1]
    ctx_mean = ctx

    init_state = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_state', activ='tanh')

    print 'Building f_init...',
    outs = [init_state, ctx]
    f_init = theano.function([x], outs, name='f_init', profile=profile)
    print 'Done'

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')
    ctx = tensor.matrix('ctx', dtype='float32')

    # if it's the first word, emb should be all zero
    emb = tensor.switch(y[:,None] < 0, 
                        tensor.alloc(0., 1, tparams['Wemb'].shape[1]), 
                        tparams['Wemb'][y])
    proj = get_layer(options['decoder'])[1](tparams, emb, options, 
                                            prefix='decoder', 
                                            mask=None, context=ctx, 
                                            one_step=True, 
                                            init_state=init_state)
    next_state = proj

    logit = get_layer('ff')[1](tparams, proj, options, prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    next_probs = tensor.nnet.softmax(logit)
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # next word probability
    print 'Building f_next..', 
    inps = [y, ctx, init_state]
    outs = [next_probs, next_sample, next_state]
    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print 'Done'

    return f_init, f_next

# generate sample
def gen_sample(tparams, f_init, f_next, x, options, trng=None, k=1, maxlen=30, 
               stochastic=True, argmax=False):
    if k > 1:
        assert not stochastic, 'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []

    ret = f_init(x)
    next_state, ctx0 = ret[0], ret[1]
    next_w = -1 * numpy.ones((1,)).astype('int64')

    for ii in xrange(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1])
        inps = [next_w, ctx, next_state]
        ret = f_next(*inps)
        next_p, next_w, next_state = ret[0], ret[1], ret[2]

        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score += next_p[0,nw]
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:,None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]
            
            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[ti])
                new_hyp_states.append(copy.copy(next_state[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score

def recon_err(f_recon_err, prepare_data, data, iterator, 
              corrupt=None, verbose=False, 
              use_preemb=False, wv_emb=None):
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 1)).astype('float32')

    n_done = 0

    for _, valid_index in iterator:
        x, mask = prepare_data([valid[0][t] for t in valid_index])
        if corrupt is not None:
            x_noise = corrupt(x)
        else:
            x_noise = copy.copy(x)
        if use_preemb:
            shp = x_noise.shape
            x_noise = wv_embs[x_noise.flatten()].reshape([shp[0], shp[1], wv_embs.shape[1]])
        pred_probs = f_recon_err(x,x_noise,mask)
        probs[valid_index] = pred_probs[:,None]

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples computed'%(n_done,n_samples)

    return probs

# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rup2'%k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up)
    
    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up, on_unused_input='ignore')

    return f_grad_shared, f_update

def adam(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup)

    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad'%k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up)

    updir = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_updir'%k) for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up, on_unused_input='ignore')

    return f_grad_shared, f_update

def sgd(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup)

    return f_grad_shared, f_update

def perplexity(f_cost, lines, worddict, options, verbose=False, wv_embs=None):
    n_lines = len(lines)
    cost = 0.
    n_words = 0.

    for i, line in enumerate(lines):
        # get array from line
        wordin = wordpunct_tokenize(line.strip())
        seq = [worddict[w] if w in worddict else 1 for w in wordin]
        seq = [s if s < options['n_words'] else 1 for s in seq]
        n_words += len(seq)+1
        x = numpy.array(seq+[0]).astype('int64').reshape([len(seq)+1,1])
        x_mask = numpy.ones((len(seq)+1,1)).astype('float32')
        if options['use_preemb']:
            shp = x.shape
            xi = wv_embs[x.flatten()].reshape([shp[0], shp[1], wv_embs.shape[1]])
        else:
            xi = x
        cost_one = f_cost(x, x_mask, xi, x_mask) * (len(seq)+1)
        cost += cost_one

        if verbose:
            print 'Sentence ', i, '/', n_lines, ' (', seq.mean(), '):', 2 ** (cost_one/len(seq)/numpy.log(2)), ', ', cost_one/len(seq)
    cost = cost / n_words
    return cost

def train(dim_word=100, # word vector dimensionality
          dim=1000, # the number of RNN units
          patience=10,
          max_epochs=5000,
          dispFreq=100,
          corruption=['_mask', '_shuffle'],
          corruption_prob=[0.1, 0.1],
          decay_c=0., 
          lrate=0.01, 
          clip_c=-1.,
          param_noise=0.,
          n_words=50000,
          maxlen=100, # maximum length of the description
          optimizer='adam', 
          batch_size = 16,
          valid_batch_size = 16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000, # save the parameters after every saveFreq updates
          encoder='gru',
          decoder='gru_cond',
          dataset='wiki',
          use_preemb=True,
          embeddings='../Files/D_medium_cbow_pdw_8B.pkl',
          dictionary='../Files/dict.pkl',
          valid_text='../Files/newsdev.tok.en',
          test_text='../Files/newstest.tok.en',
          use_dropout=False,
          reload_=False):

    # Model options
    model_options = locals().copy()

    if dictionary:
        with open(dictionary, 'rb') as f:
            worddict = pkl.load(f)
        word_idict = dict()
        for kk, vv in worddict.iteritems():
            word_idict[vv] = kk

    if use_preemb:
        assert dictionary, 'Dictionary must be provided'

        with open(embeddings, 'rb') as f:
            wv = pkl.load(f)
        wv_embs = numpy.zeros((len(worddict.keys()), len(wv.values()[0])), dtype='float32')
        for ii, vv in wv.iteritems():
            if ii in worddict:
                wv_embs[worddict[ii],:] = vv
        wv_embs = wv_embs.astype('float32')
        model_options['dim_word'] = wv_embs.shape[1]
    else:
        wv_embs = None

    # reload options
    if reload_ and os.path.exists(saveto):
        with open('%s.pkl'%saveto, 'rb') as f:
            reloaded_options = pkl.load(f)
            model_options.update(reloaded_options)

    print 'Loading data'
    load_data = get_dataset(dataset)
    train, valid, test = load_data(batch_size=batch_size)

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    tparams = init_tparams(params)
    trng, use_noise, \
          x, x_mask, x_noise, xn_mask, \
          ctx, \
          cost = \
          build_model(tparams, model_options)
    inps = [x, x_mask, x_noise, xn_mask]

    if param_noise > 0.:
        noise_update = []
        noise_tparams = OrderedDict()
        for kk, vv in tparams.iteritems():
            noise_tparams[kk] = theano.shared(vv.get_value() * 0.)
            noise_update.append((noise_tparams[kk], param_noise * trng.normal(vv.shape)))
        f_noise = theano.function([], [], updates=noise_update)
        add_update = []
        rem_update = []
        for vv, nn in zip(tparams.values(), noise_tparams.values()):
            add_update.append((vv, vv + nn))
            rem_update.append((vv, vv - nn))
        f_add_noise = theano.function([], [], updates=add_update)
        f_rem_noise = theano.function([], [], updates=rem_update)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost)
    print 'Done'

    # sentence representation
    print 'Building f_ctx...',
    f_ctx = theano.function([x_noise, xn_mask], ctx)
    print 'Done'

    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost)
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    f_grad = theano.function(inps, grads)
    print 'Done'

    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2), 
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    print 'Done'

    print 'Optimization'

    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        history_errs = list(numpy.load(saveto)['history_errs'])
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size

    # validation and test
    if valid_text:
        valid_lines = []
        with open(valid_text, 'r') as f:
            for l in f:
                valid_lines.append(l.lower())
        n_valid_lines = len(valid_lines)
    if test_text:
        test_lines = []
        with open(test_text, 'r') as f:
            for l in f:
                test_lines.append(l.lower())
        n_test_lines = len(test_lines)


    uidx = 0
    estop = False
    for eidx in xrange(max_epochs):
        n_samples = 0

        if 'start' in dir(train):
            train.start()
        for x in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            x_noise = []
            for xx_ in x:
                xn_ = xx_
                for cc, pp in zip(corruption, corruption_prob):
                    xn_ = eval(cc)(xn_, degree=pp, use_preemb=use_preemb)
                x_noise.append(xn_)
            x, x_mask, x_noise, xn_mask = prepare_data(x, x_noise, 
                                                       maxlen=maxlen, 
                                                       n_words=n_words)

            if model_options['use_preemb']:
                shp = x_noise.shape
                x_noise = wv_embs[x_noise.flatten()].reshape([shp[0], shp[1], wv_embs.shape[1]])

            if x == None:
                print 'Minibatch with zero sample under length ', maxlen
                continue

            if param_noise > 0.:
                f_noise()
                f_add_noise()
            cost = f_grad_shared(x, x_mask, x_noise, xn_mask)
            if param_noise > 0.:
                f_rem_noise()
            f_update(lrate)

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                #import ipdb; ipdb.set_trace()

                if best_p != None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, **params)
                pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))
                print 'Done'

            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                train_err = 0
                valid_err = 0
                test_err = 0
                #for _, tindex in kf:
                #    x, mask = prepare_data(train[0][train_index])
                #    train_err += (f_pred(x, mask) == train[1][tindex]).sum()
                #train_err = 1. - numpy.float32(train_err) / train[0].shape[0]

                #train_err = pred_error(f_pred, prepare_data, train, kf)
                if valid_text != None:
                    valid_err = perplexity(f_cost, valid_lines, worddict, model_options, wv_embs=wv_embs)
                if test_text != None:
                    test_err = perplexity(f_cost, test_lines, worddict, model_options, wv_embs=wv_embs)

                history_errs.append([valid_err, test_err])

                if len(history_errs) > 1:
                    if uidx == 0 or valid_err <= numpy.array(history_errs)[:,0].min():
                        best_p = unzip(tparams)
                        bad_counter = 0
                    if eidx > patience and valid_err >= numpy.array(history_errs)[:-patience,0].min():
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

                print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

        #print 'Epoch ', eidx, 'Update ', uidx, 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

        print 'Seen %d samples'%n_samples

        if estop:
            break

    if best_p is not None: 
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    train_err = 0
    valid_err = 0
    test_err = 0
    #train_err = pred_error(f_pred, prepare_data, train, kf)
    if valid_text != None:
        valid_err = perplexity(f_cost, valid_lines, worddict, model_options, wv_embs=wv_embs)
    if test_text != None:
        test_err = perplexity(f_cost, test_lines, worddict, model_options, wv_embs=wv_embs)

    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p, train_err=train_err, 
                valid_err=valid_err, test_err=test_err, history_errs=history_errs, 
                **params)

    return train_err, valid_err, test_err



if __name__ == '__main__':
    pass












    



