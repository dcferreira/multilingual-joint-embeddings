from src import config
import codecs
import os
from src.word_reps.datasets.europarl.bow import gen_repM

foreign_languages = ['ar', 'de', 'es', 'fr', 'it', 'nl', 'pb', 'pl', 'ro', 'ru', 'tr']
foreign_languages_dict = dict(zip(foreign_languages, range(len(foreign_languages))))
classes = ['art', 'arts', 'biology', 'business', 'creativity', 'culture', 'design', 'economics', 'education', 'entertainment', 'global', 'health', 'politics', 'science', 'technology']
classes_dict = dict(zip(classes, range(len(classes))))
n_train_docs = 1500
dev_set_size = 150
n_test_docs = 231

def get_counts(data):
    vocab_sizeEN = 0
    vocab_sizeDE = 0
    ENcounts = {} # nr of times a word appears
    DEcounts = {}

    for lang_nr, lang in enumerate(foreign_languages):
        for c in classes:
            for filename in sorted(os.listdir(config.TED_TRAIN_DIR + 'en-' + lang + os.sep + 'train' + os.sep + c + os.sep + 'positive')):
                with codecs.open(config.TED_TRAIN_DIR + 'en-' + lang + os.sep + 'train' + os.sep + c + os.sep + 'positive' + os.sep + filename, 'r', 'utf-8') as fileEN:
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

                            vocab_sizeEN = update_dicts(sentenceEN, vocab_sizeEN, ENcounts)
                            vocab_sizeDE = update_dicts(sentenceDE, vocab_sizeDE, DEcounts)
    print 'vocab of EN:', vocab_sizeEN
    print 'vocab of foreign:', vocab_sizeDE

    return ENcounts, DEcounts

def update_dicts(sentence, vocab_size, counts):
    for w in sentence:
        if w in counts:
            counts[w] += 1
        else:
            counts[w] = 1
            vocab_size += 1
    return vocab_size

def build_reps(data):
    print "building bag of words reps..."
    ENcounts, DEcounts = get_counts(data)
    data.EN, data.word_to_indexEN = gen_repM(ENcounts)
    data.DE, data.word_to_indexDE = gen_repM(DEcounts)
