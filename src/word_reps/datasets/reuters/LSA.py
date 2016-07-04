from src import config
import os
import numpy as np
import codecs
from src.utils import word_reps

def build_data(data, reuters_set = 'EN10000'):
    data.Z = get_Z('../../' + config.REUTERS_TRAIN_DIR + reuters_set, data.EN, data.word_to_indexEN)

def get_Z(reuters_file, repM, word_to_index):
    # input reuters directory, and matrix of representations + word to index dictionary
    # output matrix Z with reuters documents representations
    class_dict = {'C': 0., 'E': 1., 'G': 2., 'M': 3.}
    classes = ['C', 'E', 'G','M']
    nfiles = 0
    for direct in classes:
        nfiles += len(os.listdir(config.REUTERS_TEST_DIR + reuters_file + os.sep + direct)) # count nr of files


    Z = []
    n = 0

    for direct in classes:
        for filename in sorted(os.listdir(config.REUTERS_TEST_DIR + reuters_file + os.sep + direct)):
            with codecs.open(config.REUTERS_TEST_DIR + reuters_file + os.sep + direct + os.sep + filename, 'r','utf-8') as f:
                rep = np.zeros(np.shape(repM)[1]+1)
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
                    rep += word_reps.sentence_to_dense_rep(sentence, repM, word_to_index, count, class_dict[direct])
                rep[-1] = class_dict[direct]
                Z.append(rep)

    Z = np.array(Z).T

    return Z
