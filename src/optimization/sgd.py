from src import config
import math
import time
import numpy as np
import random
import pickle
import sys
import os
from src.word_reps import general
from src.utils import files


class SaveState:
    def __init__(self, inp, G, function_name_str, grad_fun, data, step0, it, adagrad, split_blocks, preproc_fun, callback_gen, fd, args):
        self.function_name_str = function_name_str
        self.grad_fun = grad_fun
        self.inp = inp
        self.build_reps = data.build_reps
        self.G = G
        self.step0 = step0
        self.it = it
        self.adagrad = adagrad
        self.split_blocks = split_blocks
        self.preproc_fun = preproc_fun
        self.callback_gen = callback_gen
        self.n_blocks = data.n_blocks
        self.args = args[1:] # remove data from args
        self.filename = os.path.abspath(fd.name)

    def save(self, filename):
        with open(filename, 'wb+') as f:
            pickle.dump(self, f, -1)

def resume(data, filename, maxit):
    with open(filename,'r') as f:
        S = pickle.load(f)
    S = S[1]
    for params in S.build_reps:
        general.build_data(data, params[0], params[1], params[2])
    function_name = files.import_file('src.optimization.functions.' + str(S.function_name_str))
    data.n_blocks = S.n_blocks
    function_name.update_solution(data, S.inp)
    S.args.insert(0, data) # prepend data on args
    fd = open(S.filename, 'ab+')
    out = sgd(S.function_name_str, S.grad_fun, S.inp, data, S.step0, maxit, S.adagrad, S.split_blocks, S.preproc_fun, S.callback_gen, fd, S.args, S.it, S.G)
    fd.close()
    return out


def sgd(function_name_str, grad_fun, x0, data, step0, it, adagrad, split_blocks, preproc_fun=False, callback_gen=False, logfile = None, args=(), init_it = 0, init_G = None):
    """
    Minimizes function for which the gradient is grad_fun, starting at x0 (usually a vector of zeros).

    grad_fun : receives inp, data, args and outputs the gradient at inp
    x0 : initialization of the vector to be optimized
    data : Data object
    step0 : initial stepsize
    it : number of iterations/epochs
    split_blocks : function which receives data, args and current block, and updates either Data or args, in order to have grad_fun only compute the subgradient in the current block
    preproc_fun : function is called before the algorithm, for any pre-processing needs. Takes data, args
    callback_gen : function which receives data, args, and outputs a function of inp, which is called after every epoch
    args : arguments to be used in the previous functions

    """
    if logfile is None:
        fd = sys.stdout
    else:
        fd = logfile
    random.seed(config.RSEED)
    np.random.seed(config.RSEED)
    config.ITE_NR = init_it
    config.FINISHED = False
    thres = 0.
    if preproc_fun:
        preproc_fun(*args, fd=fd)

    # start SGD
    if callback_gen:
        cb_fun = callback_gen(*args, fd=fd)
    else:
        def cb_fun(inp):
            pass
    inp = x0
    if init_G is None:
        G = np.zeros(len(inp))
    else:
        G = init_G
    for t in xrange(init_it, it):
        if not adagrad:
            step = step0/math.sqrt(t+1.)
        cb_fun(inp)
        init = time.clock()
        for b in xrange(data.n_blocks):
            split_blocks(b, *args, fd=fd)
            if adagrad:
                gt = grad_fun(inp, *args, fd=fd)
                mask = abs(gt) > thres
                gts = gt[mask]
                G[mask] += gts*gts
                inp[mask] -= step0 / np.sqrt(G[mask]) * gts
            else:
                inp -= step * grad_fun(inp, *args, fd=fd)
        print "--------------------------- " + str(time.clock() - init)

    config.FINISHED = True
    cb_fun(inp)
    S = SaveState(inp, G, function_name_str, grad_fun, data, step0, it, adagrad, split_blocks, preproc_fun, callback_gen, logfile, args)
    print 'Finished'
    return inp, S
