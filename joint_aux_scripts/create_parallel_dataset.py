import os
import os.path
import sys
import pdb

path_dataset = sys.argv[1] # Example: /mnt/data/corpora/ted-cldc.
#target_languages = ['de', 'es'] #, 'fr', 'it', 'pb']
target_languages = ['ar', 'de', 'es', 'fr', 'it', 'nl', 'pb', 'pl', 'ro', 'ru',
                    'tr', 'zh']
category = 'art' # Any category will work here.

source_filepaths = []
target_filepaths = []
for language in sorted(target_languages):
    pair_en_x = 'en-%s' % language
    pair_x_en = '%s-en' % language
    path = os.sep.join([path_dataset, pair_en_x, 'train', category, 'positive'])
    target_path = os.sep.join([path_dataset, pair_x_en, 'train', category,
                               'positive'])
    alternative_target_path = os.sep.join([path_dataset, pair_x_en, 'train',
                                           category, 'negative'])
    documents = [cluster for cluster in os.listdir(path) if \
                 cluster.endswith('.ted')]
    for document in sorted(documents):
        source_filepath = path + os.sep + document
        target_filepath = target_path + os.sep + document
        if not os.path.isfile(target_filepath):
            print >> sys.stderr, "Not found: %s" % target_filepath
            target_filepath = alternative_target_path + os.sep + document
            assert os.path.isfile(target_filepath), pdb.set_trace()
        source_filepaths.append(source_filepath)
        target_filepaths.append(target_filepath)

    path = os.sep.join([path_dataset, pair_en_x, 'train', category, 'negative'])
    target_path = os.sep.join([path_dataset, pair_x_en, 'train', category,
                               'negative'])
    alternative_target_path = os.sep.join([path_dataset, pair_x_en, 'train',
                                           category, 'positive'])
    documents = [cluster for cluster in os.listdir(path) if \
                 cluster.endswith('.ted')]
    #pdb.set_trace()
    for document in sorted(documents):
        source_filepath = path + os.sep + document
        target_filepath = target_path + os.sep + document
        if not os.path.isfile(target_filepath):
            print >> sys.stderr, "Not found: %s" % target_filepath
            target_filepath = alternative_target_path + os.sep + document
            assert os.path.isfile(target_filepath), pdb.set_trace()
        source_filepaths.append(source_filepath)
        target_filepaths.append(target_filepath)

f_out_source = open('ted.source', 'w')
f_out_target = open('ted.target', 'w')
source_vocabulary = {}
target_vocabulary = {}
for source_filepath, target_filepath in zip(source_filepaths, target_filepaths):
    f_source = open(source_filepath)
    f_target = open(target_filepath)
    for line_source, line_target in zip(f_source, f_target):
        f_out_source.write(line_source)
        f_out_target.write(line_target)
        source_words = line_source.rstrip('\n').split(' ')
        target_words = line_target.rstrip('\n').split(' ')
        for word in source_words:
            if word in source_vocabulary:
                source_vocabulary[word] += 1
            else:
                source_vocabulary[word] = 1
        for word in target_words:
            if word in target_vocabulary:
                target_vocabulary[word] += 1
            else:
                target_vocabulary[word] = 1

    f_source.close()
    f_target.close()

f_out_source.close()
f_out_target.close()
f_out_source_vocab = open('ted_vocab.source', 'w')
f_out_target_vocab = open('ted_vocab.target', 'w')
for word in source_vocabulary:
    f_out_source_vocab.write('%s\t%d\n' % (word, source_vocabulary[word]))
for word in target_vocabulary:
    f_out_target_vocab.write('%s\t%d\n' % (word, target_vocabulary[word]))
f_out_source_vocab.close()
f_out_target_vocab.close()
