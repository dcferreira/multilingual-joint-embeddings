from src import config
from src.utils import word_reps
from src.word_reps.datasets.europarl import general
from scipy.sparse import *
import os
import sys
import codecs

def gen_repM(counts):
    """
    Receives a dictionary with the frequency of each word, and generates bag of words representations with the thresholds defined in src.config
    """
    i = 0
    word_to_index = {}
    common_words = sorted(counts.items(), key = (lambda x: x[1]), reverse=True)[:config.COMMON_WORDS]
    for word, count in counts.items():
        if count < config.K_LINES or word in common_words or count < config.K_COLS: # no representation for this word
            del(counts[word])
        else:
            word_to_index[word] = i
            i += 1

    col = []
    row = []
    data = []
    for word, count in counts.items():
        col.append(word_to_index[word])
        row.append(word_to_index[word])
        data.append(1.)
    repM = csr_matrix(coo_matrix((data, (row, col)), shape = (len(counts), len(counts))))

    return repM, word_to_index

def build_reps(data, nr_sentences, vocab, reuters, rev=False):
    print "building bag of words reps..."
    ENcounts, DEcounts = general.get_counts(data, vocab, nr_sentences, 'bow', reuters, rev)
    data.EN, data.word_to_indexEN = gen_repM(ENcounts)
    data.DE, data.word_to_indexDE = gen_repM(DEcounts)

def build_data(data, nr_sentences, vocab, reuters, rev):
    """
    Updates data such that data.X is the matrix representation of the EN europarl -- data.X is (#ENvocabulary x nr_sentences)
    Updates data such that data.Y is the matrix representation of the DE europarl -- data.Y is (#DEvocabulary x nr_sentences)
    """
    europarlEN_file = config.ALIGNED_EN_FILE
    europarlDE_file = config.ALIGNED_DE_FILE
    colEN = []
    rowEN = []
    dataEN = []

    colDE = []
    rowDE = []
    dataDE = []
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
                word_reps.sentence_to_rep(sentenceEN, data.EN, data.word_to_indexEN, colEN, rowEN, dataEN, countEN, n)
                sentenceDE = lineDE.lower().split()
                word_reps.sentence_to_rep(sentenceDE, data.DE, data.word_to_indexDE, colDE, rowDE, dataDE, countDE, n)

                n += 1
                if nr_sentences != 0 and n >= nr_sentences:
                    break

    if rev:
        data.Y = csr_matrix(coo_matrix((dataEN, (rowEN, colEN)), shape = (data.EN.shape[1], n)))
        data.X = csr_matrix(coo_matrix((dataDE, (rowDE, colDE)), shape = (data.DE.shape[1], n)))
        EN = data.EN
        data.EN = data.DE
        data.DE = EN
        word_to_indexEN = data.word_to_indexEN
        data.word_to_indexEN = data.word_to_indexDE
        data.word_to_indexDE = word_to_indexEN
    else:
        data.X = csr_matrix(coo_matrix((dataEN, (rowEN, colEN)), shape = (data.EN.shape[1], n)))
        data.Y = csr_matrix(coo_matrix((dataDE, (rowDE, colDE)), shape = (data.DE.shape[1], n)))
        data.rowDE = rowDE
        data.colDE = colDE
