import os
from src import config
import numpy as np
from src.utils import word_reps
from scipy.sparse import *
from src.word_reps.datasets.ted.general import *
import theano

def build_parallel_data(data):
    colEN = []
    rowEN = []
    dataEN = []

    # DE means foreign language
    colDE = []
    rowDE = []
    dataDE = []

    idx = 0
    for lang_nr, lang in enumerate(foreign_languages):
        c = 'culture' # every document is in every class, so this can be any class
        for type in ['positive', 'negative']:
            for filename in sorted(os.listdir(config.TED_TRAIN_DIR + 'en-' + lang + os.sep + 'train' + os.sep + c + os.sep + type)):
                with codecs.open(config.TED_TRAIN_DIR + 'en-' + lang + os.sep + 'train' + os.sep + c + os.sep + type + os.sep + filename, 'r', 'utf-8') as fileEN:
                    if os.path.isfile(config.TED_TRAIN_DIR + lang + '-en' + os.sep + 'train' + os.sep + c + os.sep + 'positive' + os.sep + filename):
                        in_class = 'positive'
                    elif os.path.isfile(config.TED_TRAIN_DIR + lang + '-en' + os.sep + 'train' + os.sep + c + os.sep + 'negative' + os.sep + filename):
                        in_class = 'negative'
                    else:
                        raise IOError('no such file ' + config.TED_TRAIN_DIR + lang + '-en' + os.sep + 'train' + os.sep + c + os.sep + 'positive' + os.sep + filename)
                    with codecs.open(config.TED_TRAIN_DIR + lang + '-en' + os.sep + 'train' + os.sep + c + os.sep + in_class + os.sep + filename, 'r', 'utf-8') as fileDE:
                        for lineEN, lineDE in zip(fileEN, fileDE):
                            if lineEN in ['\n','\r\n'] or lineDE in ['\n','\r\n']: # ignore empty lines
                                continue
                            sentenceEN = lineEN.split()
                            sentenceDE = lineDE.split()

                            word_reps.sentence_to_rep(sentenceEN, data.EN, data.word_to_indexEN, colEN, rowEN, dataEN, False, idx)
                            word_reps.sentence_to_rep(sentenceDE, data.DE, data.word_to_indexDE, colDE, rowDE, dataDE, False, idx)
                            idx += 1

    data.X = csr_matrix(coo_matrix((dataEN, (rowEN, colEN)), shape = (data.EN.shape[1], idx)))
    data.Y = csr_matrix(coo_matrix((dataDE, (rowDE, colDE)), shape = (data.DE.shape[1], idx)))

def get_Z(data, type = 'train', source = 'en', target ='de'):
    col = []
    row = []
    values = []
    direction = source + '-' + target
    if type == 'train':
        ndocs = n_train_docs
    else:
        ndocs = n_test_docs
    labels = np.zeros((ndocs, len(classes)))
    for c in classes:
        for filename in sorted(os.listdir(config.TED_TRAIN_DIR + direction + os.sep + type + os.sep + c + os.sep + 'positive')):
            with codecs.open(config.TED_TRAIN_DIR + direction + os.sep + type + os.sep + c + os.sep + 'positive' + os.sep + filename, 'r', 'utf-8') as fd:
                idx = int(filename.split('.')[0])
                if type == 'test':
                    idx -= n_train_docs
                n_sent = 0
                for line in fd:
                    n_sent += 1
                fd.seek(0)
                for line in fd:
                    sentence = line.split()
                    if source == 'en':
                        word_reps.sentence_to_rep(sentence, data.EN, data.word_to_indexEN, col, row, values, n_sent, idx)
                    else:
                        word_reps.sentence_to_rep(sentence, data.DE, data.word_to_indexDE, col, row, values, n_sent, idx)
                labels[idx, classes_dict[c]] += 1.

    labels = np.nan_to_num(labels.T / labels.sum(1)).T.astype(theano.config.floatX)
    if source == 'en':
        Z = csr_matrix(coo_matrix((values, (row, col)), shape = (data.EN.shape[1], ndocs)), dtype = theano.config.floatX)
    else:
        Z = csr_matrix(coo_matrix((values, (row, col)), shape = (data.DE.shape[1], ndocs)), dtype = theano.config.floatX)
    return labels, Z

def build_classification_data(data):
    data.Z_classes, data.Z = get_Z(data, 'train')
    data.ZENdev_classes = data.Z_classes[-dev_set_size:]
    data.ZENdev = data.Z[:,-dev_set_size:]
    data.Z_classes = data.Z_classes[:-dev_set_size]
    data.Z = data.Z[:, : -dev_set_size]

    data.ZENtest_classes, data.ZENtest = get_Z(data, 'test')

    data.ZDEtest = []
    data.ZDEtest_classes = []
    data.ZDEdev = []
    data.ZDEdev_classes = []
    for lang in foreign_languages:
        ZDEtest_classes, ZDEtest = get_Z(data, 'test', lang, 'en')
        ZDEdev_classes, ZDEdev = get_Z(data, 'train', lang, 'en')
        ZDEdev_classes = ZDEdev_classes[- dev_set_size :]
        ZDEdev = ZDEdev[:, - dev_set_size :]

        data.ZDEtest.append(ZDEtest)
        data.ZDEtest_classes.append(ZDEtest_classes)
        data.ZDEdev.append(ZDEdev)
        data.ZDEdev_classes.append(ZDEdev_classes)

def build_data(data):
    build_parallel_data(data)
    build_classification_data(data)

