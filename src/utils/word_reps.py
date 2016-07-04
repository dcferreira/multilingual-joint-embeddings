from src import config
import numpy as np
import codecs

def count_words(line, word_to_index):
    count = 0.
    sentence = line.lower().split()
    for word in sentence:
        if word in word_to_index:
            count += 1.
    return count

def sentence_to_rep(sentence, repM, word_to_index, col, row, data, count, n):
    for word in sentence:
        if word in word_to_index:
            col.append(n)
            row.append(word_to_index[word])
            if count > 0: # there is some averaging
                data.append(1./count)
            else: # no averaging, only sum of words
                data.append(1)


def sentence_to_dense_rep(sentence, repM, word_to_index, count, class_index = None):
    if class_index is None:
        rep = np.zeros(repM.shape[1])
    else:
        rep = np.zeros(repM.shape[1]+1)
    for word in sentence:
        if word in word_to_index:
            rep[:repM.shape[1]] += repM[word_to_index[word]]
    if count > 0: # there is some averaging
        rep /= float(count)
    if class_index is not None:
        rep[repM.shape[1]] = class_index
    return rep

def file_to_embeddings(file, word_to_index, dims, ted = False):
    print 'Reading embeddings from', file
    with open(file, 'r') as fd:
        line = fd.readline() # ignore first line, with comment
        out = np.zeros((len(word_to_index), dims))

        for line in fd:
            sline = line.strip('\n').split(' ')
            word_to_test = sline[0] + '_en' if ted else sline[0]
            if word_to_test in word_to_index:
                out[word_to_index[word_to_test]] = np.fromstring(' '.join(sline[1:(dims+1)]), sep = ' ')
    print 'Finished reading embeddings'
    return out

def embeddings_to_file(embeddings, word_to_index, output_file):
    '''
    :param embeddings:
    :param word_to_index:
    :param output_file:
    :return:
    '''
    print 'Writing embeddings to', output_file
    with codecs.open(output_file, 'w+', 'utf-8') as fd:
        for word, idx in word_to_index.items():
            fd.write(word + ' ')
            fd.write(' '.join([str(n) for n in embeddings[idx]]))
            fd.write('\n')
    print 'Finished writing embeddings'
