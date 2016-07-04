from src.word_reps.datasets.europarl import general
from src import config
import math
from scipy.sparse import *
from scipy.sparse import linalg as slinalg
import numpy as np
from src.utils import word_reps
import codecs
from src.word_reps.datasets.europarl import bow

def gen_mat(counts, freqs, total_counts, type, feat_thresh = config.K_COLS):
    l = 0
    c = 0
    word_to_index = {}
    word_to_feat = {}
    common_words = sorted(counts.items(), key = (lambda x: x[1]), reverse=True)[:config.COMMON_WORDS]
    for word, count in counts.items():
        if count < config.K_LINES or (word in common_words and config.K_LINES != 0 ) or count < config.K_COLS: # no representation for this word
            del(counts[word])
        else:
            if count >= feat_thresh and not word in common_words: # word works as feature
                word_to_feat[word] = c
                c += 1
            word_to_index[word] = l
            l += 1

    col = []
    row = []
    data = []
    for keypair, count in freqs.items():
        if keypair[0] in word_to_index and keypair[1] in word_to_feat:
            row.append(word_to_index[keypair[0]])
            col.append(word_to_feat[keypair[1]])
            if type == 'pmi':
                data.append(math.log((float(count) * l) / (counts[keypair[0]] * counts[keypair[1]]))) # log[f(i,j) * N / (f(i) * f(j))]
            elif type == 'RelFreq':
                data.append(float(count) / float(counts[keypair[0]]))
            elif type == 'RelFreqDif':
                if keypair[1] == keypair[0]:
                    data.append(float(count) / float(counts[keypair[0]]))
                else:
                    data.append(- float(count) / float(counts[keypair[0]]))
            elif type == 'AbsFreqDif':
                if keypair[1] == keypair[0]:
                    data.append(float(count))
                else:
                    data.append(-float(count))

    return csr_matrix(coo_matrix((data, (row, col)), shape = (l, c)))

def gen_repM(counts, freqs, total_counts):
    """
    Receives a dictionary with the frequency of each word, and generates LSA representations with the thresholds defined in src.config
    """
    M = gen_mat(counts, freqs, total_counts, 'pmi')

    U, D, _ = slinalg.svds(M, config.NCOLS)
    repM = U.dot(np.diag(D**config.DIAG_EXP))

    return repM, word_to_index


def build_reps(data, nr_sentences, vocab, reuters, rev=False):
    print "building LSA reps..."
    ENcounts, DEcounts, ENfreqs, DEfreqs, total_countsEN, total_countsDE = general.get_counts(data, vocab, nr_sentences, 'LSA', reuters)
    data.EN, data.word_to_indexEN = gen_repM(ENcounts, ENfreqs, total_countsEN)
    data.DE, data.word_to_indexDE = gen_repM(DEcounts, DEfreqs, total_countsDE)

def build_data(data, nr_sentences, vocab, reuters, rev):
    """
    Updates data such that data.X is the matrix representation of the EN europarl -- data.X is (#ENvocabulary x nr_sentences)
    Updates data such that data.Y is the matrix representation of the DE europarl -- data.Y is (#DEvocabulary x nr_sentences)
    """
    europarlEN_file = config.ALIGNED_EN_FILE
    europarlDE_file = config.ALIGNED_DE_FILE

    X = []
    Y = []
    with codecs.open(europarlEN_file, 'r','utf-8') as fileEN:
        with codecs.open(europarlDE_file, 'r','utf-8') as fileDE:
            n = 0 # number of sentences already read
            for lineEN, lineDE in zip(fileEN, fileDE):
                if lineEN in ['\n','\r\n'] or lineDE in ['\n','\r\n']: # ignore empty lines
                    continue
                if config.AVG:
                    countEN = word_reps.count_words(lineEN, data.word_to_indexEN)
                    countDE = word_reps.count_words(lineDE, data.word_to_indexDE)
                else:
                    countEN = countDE = False

                sentenceEN = lineEN.lower().split()
                X.append(word_reps.sentence_to_dense_rep(sentenceEN, data.EN, data.word_to_indexEN, countEN))
                sentenceDE = lineDE.lower().split()
                Y.append(word_reps.sentence_to_dense_rep(sentenceDE, data.DE, data.word_to_indexDE, countDE))

                n += 1
                if nr_sentences != 0 and n >= nr_sentences:
                    break
    if rev:
        data.Y = np.array(X).T
        data.X = np.array(Y).T
        EN = data.EN
        data.EN = data.DE
        data.DE = EN
        word_to_indexEN = data.word_to_indexEN
        data.word_to_indexEN = data.word_to_indexDE
        data.word_to_indexDE = word_to_indexEN
    else:
        data.X = np.array(X).T
        data.Y = np.array(Y).T
