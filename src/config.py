import os

ALIGNED_EN_FILE = 'data' + os.sep + 'europarl' + os.sep + 'europarl-v7.de-en.tokenized.en'
ALIGNED_DE_FILE = 'data' + os.sep + 'europarl' + os.sep + 'europarl-v7.de-en.tokenized.de'
REUTERS_TRAIN_DIR = 'data' + os.sep + 'reuters' + os.sep + 'train' + os.sep
REUTERS_TEST_DIR = 'data' + os.sep + 'reuters' + os.sep + 'test' + os.sep 
AVG = True # if true, representation of a sentence is the average of the words, if false, it's just the sum
SENT_AVG = True # if true, representation of a document is average of sentences
K_COLS = 10 # minimum frequency a word needs to have to be a feature in bag of words representation
K_LINES = 10 # minimum frequency a word needs to have to have a representation in bag of words -- if 0, every found word has representation
WINDOW_SIZE = 4 # window size around word in which to count co-occurencies
COMMON_WORDS = 100 # the most common words won't be a feature
NCOLS = 640 # nr of features to use in LSA
RSEED = 418318 # seed for random
ZSEED = 18373 # seed for shuffling Z (same as used before refactor)
ITE_PRINTS = 50 # print callback funtion each ITE_PRINTS iterations
MAXENT_DEBUG = True # flag for debugging maxent
OPTIMIZATION = False # flag for optimization in the gradient (uses a lot of memory)
PRINTS = False # flag for new_cost/new_grad prints
PRINT_TIMES = True # print times througout the code
DEBUG = False # debug flag

#LSA options
SYMMETRIC_PMI = True # PMI computed as symmetric or not
DIAG_EXP = 1. # exponent on the diagonal of the SVD

ALT_PQ = False

P_GLOBAL = 0
Q_GLOBAL = 0
ITE_NR = 0
FINISHED = False

COMPUTE_LOSS = True

intX = 'int64'
TSEED = 418318 # seed for theano random stream

COOC_ONLY_ONE = False
COOC_THRESH = 10

TED_TRAIN_DIR = 'data' + os.sep + 'ted-cldc' + os.sep
TEST_THRESH = 0.18

BIN_MULTI = False
BIN_TEST_THRESH = 0.5
