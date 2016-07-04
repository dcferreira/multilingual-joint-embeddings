import theano
import theano.tensor as T
from theano import sparse
import numpy as np
from src import config

eps = 1e-16
def norm(mat, axis=None, l=2.):
    if l == 2.:
        return T.cast(T.pow(T.sum(T.maximum(mat**l, eps), axis=axis), 1./l), theano.config.floatX)
    else:
        return T.cast(T.pow(T.sum(T.maximum(abs(mat)**l, eps), axis=axis), 1./l), theano.config.floatX)

def random_indices(sdata, srng):
    return srng.random_integers(size=(sdata.X.shape[1]), high=sdata.Yfull.shape[1]-1)

def get_theano_idx_vals(sent_idx, sent_vals):
    max_len = len(max(sent_idx, key=lambda x: len(x)))
    for i in xrange(len(sent_idx)): # pad with zeros
        sent_idx[i] += [0] * (max_len - len(sent_idx[i]))
        sent_vals[i] += [0] * (max_len - len(sent_vals[i]))

    _sent_idx = theano.shared(
        value=np.array(sent_idx, dtype='int32'),
        name='sent_idx'
    )
    _sent_vals = theano.shared(
        value=np.array(sent_vals, dtype='int32'),
        name='sent_vals'
    )

    return _sent_idx, _sent_vals


def get_functions(sdata, n_blocks):
    functions = {}

    XP = sparse.basic.dot(sdata.X, sdata.P)
    YQ = sparse.basic.dot(sdata.Y, sdata.Q)
    dif = XP - YQ
    functions['--quadratic'] = (1. / sdata.data.X.shape[1]) * norm( dif )**2
    functions['--coocP'] = (1. / sdata.data.P.shape[1]) * norm( sparse.basic.dot(sdata.Xcooc, sdata.P))**2
    functions['--coocQ'] = (1. / sdata.data.Q.shape[1]) * norm( sparse.basic.dot(sdata.Ycooc, sdata.Q))**2
    functions['--l1'] = (1. / sdata.data.X.shape[1]) * norm( dif , axis=None, l=1.)

    prob = T.nnet.softmax(T.dot(sparse.basic.dot(sdata.Z.T, sdata.P), sdata.V))
    try:
        functions['--loss'] = - (1. / sdata.data.Z.shape[1]) * T.sum( T.log(prob)[T.arange(sdata.Z_classes.shape[0]), sdata.Z_classes])
    except TypeError:
        multiprob = T.nnet.categorical_crossentropy(prob, sdata.Z_classes)
        functions['--multiloss'] = (1. / sdata.data.Z.shape[1]) * T.sum( multiprob )

        gold_labels = T.switch(T.gt(sdata.Z_classes, 0.), 1., 0.)
        numerators = [T.exp(T.dot(sparse.basic.dot(sdata.Z.T, sdata.P), sdata.V[T.arange(sdata.V.shape[0]), c])) for c in range(15)]
        binprob = [numerators[c] / (1 + numerators[c]) for c in range(15)]
        probs = [gold_labels[T.arange(gold_labels.shape[0]), c] * T.log(binprob[c]) + \
                 (1 - gold_labels[T.arange(gold_labels.shape[0]), c]) * T.log(1 - binprob[c]) \
                 for c in range(15)]
        functions['--multibinloss'] = (1. / sdata.data.Z.shape[1]) * (- T.sum(probs))

    if hasattr(sdata.data, 'glove_vecs'):
        functions['--glovediff'] = norm(sdata.P - sdata.data.glove_vecs)
    functions['--regP'] = (1. / n_blocks) * norm(sdata.P)**2
    functions['--regQ'] = (1. / n_blocks) * norm(sdata.Q)**2

    if not sdata.data.negex_nr is None:
        YQfull = sparse.basic.dot(sdata.Yfull.T, sdata.Q)
        l = [T.maximum(eps, sdata.data.negex_margin + norm(dif, 1) - norm(XP - YQfull[sdata.Ynegex[i]], 1)) for i in xrange(sdata.data.negex_nr)]
        functions['--negative_sampling'] = (1. / sdata.data.Z.shape[1]) * T.sum(sum(l))

    sdata.functions = functions

    return functions

def cost(sdata, params, n_blocks):
    f = []
    functions = get_functions(sdata, n_blocks)
    for key, val in params.items():
        if key in functions:
            f.append(val * functions[key])

    return sum(f)
