import os
import os.path
import sys
import numpy as np
import pdb

def load_embeddings(embeddings_filepath, dimension):
    skip_first_line = False
    f = open(embeddings_filepath)
    # Skip first line.
    if skip_first_line:
        print >> sys.stderr, 'Skipping first line...'
        f.readline()
    num_words = 0
    for line in f:
        num_words += 1
    f.close()
    f = open(embeddings_filepath)
    # Skip first line.
    if skip_first_line:
        f.readline()
    E = np.zeros((num_words, dimension))
    vocab = []
    wid = 0
    for line in f:
        line = line.rstrip('\n')
        #fields = line.split('\t')
        fields = line.split(' ')
        word = fields[0]
        #fields = [word] + fields[1].split(' ')
        vocab.append(word)
        embedding = np.array([float(val) for val in fields[1:]])
        assert embedding.shape[0] == dimension, pdb.set_trace()
        E[wid, :] = embedding
        wid += 1
    return E, vocab

def compute_document_embedding(d, E, vocab, allow_unknowns=False):
    v = np.zeros(E.shape[1])
    for word in d:
        if word not in vocab:
            if allow_unknowns:
                continue
            else:
                assert word in vocab, pdb.set_trace()
        wid = vocab[word]
        v += d[word] * E[wid, :]
    return v

if __name__ == '__main__':
    path_dataset = sys.argv[1] # Example: /mnt/data/corpora/ted-cldc.
    source_embeddings_filepath = sys.argv[2]
    target_embeddings_filepath = sys.argv[3]

    dimension = 300

    P, source_words = \
        load_embeddings(source_embeddings_filepath, dimension)
    Q, target_words = \
        load_embeddings(target_embeddings_filepath, dimension)
    source_vocab = dict(zip(source_words, range(len(source_words))))
    target_vocab = dict(zip(target_words, range(len(target_words))))

    #target_languages = ['de', 'es'] #, 'fr', 'it', 'pb']
    target_languages = ['ar', 'de', 'es', 'fr', 'it', 'nl', 'pb', 'pl', 'ro', 'ru',
                        'tr', 'zh']

    categories = ['art', 'arts', 'biology', 'business', 'creativity', 'culture',
                  'design', 'economics', 'education', 'entertainment', 'global',
                  'health', 'politics', 'science', 'technology']

    source_filepaths = []
    target_filepaths = []
    for language in sorted(target_languages):
        pair_en_x = 'en-%s' % language
        pair_x_en = '%s-en' % language
        for pair in [pair_en_x, pair_x_en]:
            for partition in ['train', 'test']:
                filepath_out = 'ted.%s.%s' % (pair, partition)
                filepath_out_bow = 'ted.bow.%s.%s' % (pair, partition)
                f_out = open(filepath_out, 'w')
                f_out_bow = open(filepath_out_bow, 'w')
                document_labels = {}
                for l, category in enumerate(categories):
                    path = os.sep.join([path_dataset, pair, partition, category,
                                        'positive'])
                    documents = [cluster for cluster in os.listdir(path) if \
                                 cluster.endswith('.ted')]
                    for document in sorted(documents):
                        if document not in document_labels:
                            document_labels[document] = [l]
                        else:
                            document_labels[document].append(l)
                for document in document_labels:
                    category = categories[document_labels[document][0]]
                    path = os.sep.join([path_dataset, pair, partition, category,
                                        'positive'])
                    filepath = path + os.sep + document
                    f = open(filepath)
                    d = {}
                    num_sentences = 0
                    for line in f:
                        words = line.rstrip('\n').split(' ')
                        for word in words:
                            if word in d:
                                d[word] += 1.
                            else:
                                d[word] = 1.
                        num_sentences += 1
                    for word in d:
                        d[word] /= float(num_sentences)

                    allow_unknowns = (partition != 'train')
                    if pair == pair_en_x:
                        v = compute_document_embedding(d, P, source_vocab,
                                                       allow_unknowns=allow_unknowns)
                    else:
                        v = compute_document_embedding(d, Q, target_vocab,
                                                       allow_unknowns=allow_unknowns)

                    labels = document_labels[document]
                    label_field = ','.join([str(l) for l in labels])
                    doc_field = ' '.join([':'.join([str(1+k), str(v[k])]) for k in xrange(dimension)])
                    #label_field = ','.join([categories[l] for l in labels])
                    #doc_field = ' '.join([':'.join([word, str(d[word])]) for word in d])
                    f_out.write(label_field + '\t' + doc_field + '\n')
                    if pair == pair_en_x:
                        vocab = source_vocab
                    else:
                        vocab = target_vocab
                    doc_bow_field = ' '.join([':'.join([str(1+vocab[word]), str(d[word])]) for word in d if word in vocab])
                    f_out_bow.write(label_field + '\t' + doc_bow_field + '\n')

                f_out.close()
                f_out_bow.close()
