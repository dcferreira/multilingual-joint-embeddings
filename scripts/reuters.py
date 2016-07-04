import os
import os.path
import sys, getopt
import time
import pickle
import pdb
import numpy as np
import inspect
from src.data import data as data_struct
from src.word_reps import general
from src import config
import theano
import theano.tensor as T
from theano import sparse
import math
from src.word_reps.datasets.reuters import bow
from src.data import symbdata
import timeit
from scipy.sparse import *
from src.optimization.functions import theano_functions


STDOUT = False
NIT = 1000
SKIPPED = []
config.AVG = False
config.SENT_AVG = True
config.K_COLS = 0
config.K_LINES = 0
config.COMMON_WORDS = 0

VERSION = 'v1'
DIMS = 4
RANDPQ = False
N_SENT = 500000
TRAIN_SET = 'EN1000_train_valid'
NRNEGEX = 10
MNEGEX = float(DIMS)
BLOCKSIZE = 100
LEARNING_RATE = 1.

config.ITE_PRINTS = 1

## PARSE ARGS
def parse_args():
    global TRAIN_SET
    global N_SENT
    global DIMS
    global RANDPQ
    global BLOCKSIZE
    global MNEGEX
    global NRNEGEX
    global NIT
    global STDOUT
    global LEARNING_RATE
    global PATHS
    global FUNCS
    global FILE
    global rev
    global params_values
    global PFILE
    global SPLIT
    SPLIT = False
    global EMB_NIT
    EMB_NIT = None
    global SAVE
    SAVE = False
    global COOC
    COOC = False
    global COOC_TYPE
    COOC_TYPE = 'AbsFreqDif'
    global CFILE
    global NEG_SAMPLING_R
    NEG_SAMPLING_R = False
    
    parameters = 'quadratic= loss= regP= regQ= negative_sampling= splitloss= l1= coocP= coocQ= negative_sampling_r='
    possible_args = (parameters + ' rev trainset= nsent= nprints= ndims= randPQ= blocksize= nit= stdout nrnegex= mnegex= adagrad= embnit= callbackite= saveemb cooctype=').split()
    parameters = parameters.split()
    try:
        opts, args = getopt.getopt(sys.argv[1:], '', possible_args)
    except getopt.GetoptError as err:    
        print str(err)
        sys.exit(0)

    rev = False
    params_values = {}
    for o, a in opts:
        if o[2:]+'=' in parameters:
            params_values[o] = float(a)
            if o == '--splitloss':
                SPLIT = True
            if o == '--coocP':
                COOC = True
            if o == '--negative_sampling_r':
                NEG_SAMPLING_R = True
        elif o == '--rev':
            rev = True
            if TRAIN_SET == 'EN1000_train_valid':
                TRAIN_SET = 'DE1000_train_valid'
        elif o =='--trainset':
            TRAIN_SET = a
        elif o == '--nsent':
            N_SENT = int(a)
        elif o == '--nprints':
            config.ITE_PRINTS = int(a)
        elif o == '--ndims':
            DIMS = int(a)
        elif o == '--randPQ':
            try:
                RANDPQ = int(a)
            except ValueError:
                RANDPQ = True
        elif o == '--blocksize':
            BLOCKSIZE = int(a)
        elif o == '--nit':
            NIT = int(a)
        elif o == '--embnit':
            EMB_NIT = int(a)
        elif o == '--callbackite':
            config.ITE_PRINTS = int(a)
        elif o == '--stdout':
            STDOUT = True
        elif o == '--saveemb':
            SAVE = True
        elif o == '--nrnegex':
            NRNEGEX = int(a)
        elif o == '--mnegex':
            MNEGEX = float(a)
        elif o == '--adagrad':
            LEARNING_RATE = float(a)
        elif o == '--cooctype':
            COOC_TYPE = a

    if EMB_NIT is None:
        EMB_NIT = NIT


    FUNCS = '_'.join( [key[2:] for key, val in sorted(params_values.items())] )
    if rev:
        PATH = 'Tests/' + VERSION + '/DE_EN/' + FUNCS + '/dims' + str(DIMS) + '/randpq' + str(RANDPQ) + '/nsent' + str(N_SENT) + '/trainset' + TRAIN_SET + '/blocksize' + str(BLOCKSIZE) + '/' + 'adagrad' + str(LEARNING_RATE) + '/'
    else:
        PATH = 'Tests/' + VERSION + '/EN_DE/' + FUNCS + '/dims' + str(DIMS) + '/randpq' + str(RANDPQ) + '/nsent' + str(N_SENT) + '/trainset' + TRAIN_SET + '/blocksize' + str(BLOCKSIZE) + '/' + 'adagrad' + str(LEARNING_RATE) + '/'
    if not os.path.isdir(PATH):
        os.makedirs(PATH)
    FILE = PATH + '_'.join( [key[2:] + str(val) for key, val in sorted(params_values.items())] )
    ## ADD FUNCTION SPECIFIC PARAMS
    if '--negative_sampling' in params_values or '--negative_sampling_r' in params_values:
        FILE += '_nrnegex' + str(NRNEGEX) + '_mnegex' + str(MNEGEX)
    if '--coocP' in params_values:
        FILE += '_cooctypePAR' + COOC_TYPE
    FILE += '.log'
    print FILE

    ## PICKLE FILE (for storing embeddings)
    PFILE = PATH + '_'.join( [key[2:] + str(val) for key, val in sorted(params_values.items()) if key != '--splitloss'] )
    if '--negative_sampling' in params_values or '--negative_sampling_r' in params_values:
        PFILE += '_nrnegex' + str(NRNEGEX) + '_mnegex' + str(MNEGEX)
    print PFILE

    ## PICKLE FILE (for cooc matrices)
    if rev:
        CFILE = 'Tests/' + VERSION + '/DE_EN/'
    else:
        CFILE = 'Tests/' + VERSION + '/EN_DE/'
    CFILE += 'cooc' + COOC_TYPE


## OPEN LOG FILE
def get_file(filename, stdoutQ):
    if stdoutQ:
        fd = sys.stdout
    else:
        if os.path.exists(filename):
            print "File exists!"
            sys.exit(0)
        fd = open(filename, 'wb+')
    return fd


## LOAD DATA
def load_data():
    mydata = data_struct.Data()
    general.build_data(mydata, 'bow', 'europarl', (N_SENT, 'union', TRAIN_SET, rev))
    general.build_data(mydata, 'bow', 'reuters', [TRAIN_SET])
    if RANDPQ == True: # seed is default
        mydata.build_randPQ(DIMS, 0.1)
    elif RANDPQ == False: # no random
        mydata.P = np.zeros((np.shape(mydata.X)[0], DIMS))
        mydata.Q = np.zeros((np.shape(mydata.Y)[0], DIMS))
    else: # seed is RANDPQ
        mydata.build_randPQ(DIMS, 0.1, seed = RANDPQ)
    if DIMS > 4:
        mydata.build_randV()
    else:
        mydata.V = np.identity(4, dtype=theano.config.floatX)
    mydata.n_blocks = np.shape(mydata.Z)[1]/BLOCKSIZE
    mydata.build_negative_examples(NRNEGEX, MNEGEX)

    sdata = symbdata.SymbData(mydata)

    if SPLIT:
        if not os.path.isfile(PFILE + '.P') or not os.path.isfile(PFILE + '.Q'):
            mydata.P, mydata.Q = create_embeddings(mydata, sdata)
        else:
            with open(PFILE + '.P', 'r') as f:
                mydata.P = pickle.load(f)
                sdata.P.set_value(mydata.P)
            with open(PFILE + '.Q', 'r') as f:
                mydata.Q = pickle.load(f)
                sdata.Q.set_value(mydata.Q)
    if COOC:
        if not os.path.isfile(CFILE + '.Xcooc') or not os.path.isfile(CFILE + '.Ycooc'):
            if rev:
                mydata.build_cooc_mat('union', 500000, TRAIN_SET, '../test/en', rev, COOC_TYPE)
            else:
                mydata.build_cooc_mat('union', 500000, TRAIN_SET, '../test/de', rev, COOC_TYPE)
            with open(CFILE + '.Xcooc', 'wb+') as f:
                pickle.dump(mydata.Xcooc, f, -1)
            with open(CFILE + '.Ycooc', 'wb+') as f:
                pickle.dump(mydata.Ycooc, f, -1)
        else:
            with open(CFILE + '.Xcooc', 'r') as f:
                mydata.Xcooc = pickle.load(f)
            with open(CFILE + '.Ycooc', 'r') as f:
                mydata.Ycooc = pickle.load(f)
    
    return mydata, sdata

def create_embeddings(mydata, sdata, save=False):
    cost, train_model, _ = load_funcs(mydata, sdata, None)

    sdata.P.set_value(np.array(np.random.randn(mydata.P.shape[0], mydata.P.shape[1])*0.1, dtype=theano.config.floatX))
    sdata.Q.set_value(np.array(np.random.randn(mydata.Q.shape[0], mydata.Q.shape[1])*0.1, dtype=theano.config.floatX))
    P = sdata.P.get_value()
    Q = sdata.Q.get_value()

    f = open(PFILE + '.pq', 'wb+')
    for i in xrange(EMB_NIT):
        start = timeit.default_timer()
        for b in xrange(mydata.n_blocks):
            if NEG_SAMPLING_R:
                sdata.data.build_negative_examples(NRNEGEX, MNEGEX, None)
            train_model(b, sdata.data.Ynegex)
        print "Iteration:", i
        print >> f, "Iteration:", i
        print "train:", timeit.default_timer() - start
        print >> f, "P:", np.linalg.norm(P - sdata.P.get_value())
        print >> f, "Q:", np.linalg.norm(Q - sdata.Q.get_value())
        P = sdata.P.get_value()
        Q = sdata.Q.get_value()
    f.close()

    if save:
        with open(PFILE + '.P', 'wb+') as f:
            pickle.dump(sdata.P.get_value(), f, -1)
        with open(PFILE + '.Q', 'wb+') as f:
            pickle.dump(sdata.Q.get_value(), f, -1)
    return sdata.P.get_value(), sdata.Q.get_value()
    

## OPTIMIZE
def split_loss(sdata, reg, learning_rate, fd):
    _, _, _ = load_funcs(sdata.data, sdata, None)
    train_model = sdata.adagradV(reg, learning_rate)
    cb = sdata.callback_gen(sdata.split_cost, sdata.data, params_values, fd, rev, True)
    return train_model, cb

def load_funcs(mydata, sdata, fd):
    cost = theano_functions.cost(sdata, params_values, mydata.n_blocks)
    train_model = sdata.adagrad(cost, LEARNING_RATE)
    cb = None
    if fd is not None:
        cb = sdata.callback_gen(cost, mydata, params_values, fd, rev)

    return cost, train_model, cb

def optimize(sdata, train_model, cb, n_blocks, fd):
    start1 = timeit.default_timer()
    for i in xrange(NIT):
        start2 = timeit.default_timer()
        if i % config.ITE_PRINTS == 0:
            cb()
        start3 = timeit.default_timer()
        for b in xrange(n_blocks):
            if NEG_SAMPLING_R:
                sdata.data.build_negative_examples(NRNEGEX, MNEGEX, None)
            train_model(b, sdata.data.Ynegex)
        print "train:", timeit.default_timer() - start3, "callback:", start3 - start2 

    cb()
    print timeit.default_timer() - start1
    print '------------- best ENdev:'
    print config.ENdev_str
    print '\n------------- best DEdev:'
    print config.DEdev_str

    print >> fd, timeit.default_timer() - start1
    print >> fd, '------------- best ENdev:'
    print >> fd, config.ENdev_str
    print >> fd, '\n------------- best DEdev:'
    print >> fd, config.DEdev_str


def save_embeddings(mydata, sdata):
    with open(FILE[:-4] + '.embeddings', 'wb+') as f:
        if rev:
            for word, idx in mydata.word_to_indexDE.items():
                f.write(word.encode('utf8') + ' ')
                for i in xrange(sdata.Q.get_value().shape[1]):
                    f.write(str(sdata.Q.get_value()[idx, i]) + ' ')
                f.write('\n')
        else:
            for word, idx in mydata.word_to_indexEN.items():
                f.write(word.encode('utf8') + ' ')
                for i in xrange(sdata.P.get_value().shape[1]):
                    f.write(str(sdata.P.get_value()[idx, i]) + ' ')
                f.write('\n')

def main():
    parse_args()
    fd = get_file(FILE, STDOUT)
    mydata, sdata = load_data()
    if SPLIT:
        train_model, cb = split_loss(sdata, params_values['--splitloss'], LEARNING_RATE, fd)
    else:
        cost, train_model, cb = load_funcs(mydata, sdata, fd)
    optimize(sdata, train_model, cb, mydata.n_blocks, fd)
    fd.close()
    if SAVE:
        save_embeddings(mydata, sdata)

if __name__ == '__main__':
    main()
