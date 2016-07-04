import sys
import numpy as np
import pdb

def find_closest(p, Q, num_neighbours):
    v = Q - p
    distances = (v*v).sum(1)
    closest = distances.argsort()[:num_neighbours]
    return closest, distances[closest]

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

if __name__ == "__main__":
    source_embeddings_filepath = sys.argv[1]
    target_embeddings_filepath = sys.argv[2]

    dimension = 300

    P, source_words = \
        load_embeddings(source_embeddings_filepath, dimension)
    Q, target_words = \
        load_embeddings(target_embeddings_filepath, dimension)
    source_vocab = dict(zip(source_words, range(len(source_words))))
    target_vocab = dict(zip(target_words, range(len(target_words))))

    num_neighbours = 20
    while True:
        word = raw_input('Type an English word: ')
        words = word.split(' ')
        if len(words) == 3: # A - B + C
            word1 = words[0] + '_en'
            word2 = words[1] + '_en'
            word3 = words[2] + '_en'
            if word1 not in source_vocab or word2 not in source_vocab or \
               word3 not in source_vocab:
                continue
            p = P[source_vocab[word1]] - P[source_vocab[word2]] + \
                P[source_vocab[word3]]
        else:
            word += '_en'
            if word not in source_vocab:
                continue
            wid = source_vocab[word]
            p = P[wid, :]
        closest, distances = find_closest(p, Q, num_neighbours)
        for tid, d in zip(closest, distances):
            print '%s: %f' % (target_words[tid], d)
        print
