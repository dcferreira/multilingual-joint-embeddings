from src.utils import word_reps
from src import config
import os
import numpy as np
import codecs
from scipy.sparse import *

def build_data(data, reuters_set = 'EN10000'):
    data.Z = get_Z(reuters_set, data.EN, data.word_to_indexEN, train=True)

def get_Z(reuters_file, repM, word_to_index, train=False):
    # input reuters directory, and matrix of representations + word to index dictionary
    # output matrix Z with reuters documents representations
    class_dict = {'C': 0, 'E': 1, 'G': 2, 'M': 3}
    classes = ['C', 'E', 'G','M']
    if train:
        dirname = config.REUTERS_TRAIN_DIR + reuters_file
    else:
        dirname = config.REUTERS_TEST_DIR + reuters_file
    nfiles = 0
    for direct in classes:
        nfiles += len(os.listdir(dirname + os.sep + direct)) # count nr of files

    n = 0

    row = []
    col = []
    data = []
    for direct in classes:
        for filename in sorted(os.listdir(dirname + os.sep + direct)):
            with codecs.open(dirname + os.sep + direct + os.sep + filename, 'r','utf-8') as f:
                if config.AVG or config.SENT_AVG:
                    count = 0.
                    for line in f:
                        if config.SENT_AVG:
                            count += 1
                        else:
                            count += word_reps.count_words(line, word_to_index) # count words with representation
                    f.seek(0,0) # restart file
                else:
                    count = False
                for line in f:
                    sentence = line.lower().split()
                    word_reps.sentence_to_rep(sentence, repM, word_to_index, col, row, data, count, n)
                col.append(n)
                row.append(np.shape(repM)[1])
                data.append(class_dict[direct])
                n += 1

    Z = csr_matrix(coo_matrix((data, (row, col)), shape = (np.shape(repM)[1]+1, nfiles)))

    return Z
