from src import config
import codecs
import os

def get_counts(data, vocab, nr_of_sentences, method, reuters, rev = False):
    ''' receives method of building vocabulary ('union', 'intersection' or 'europarl'), type of representation ('bow' or 'LSA'), EN europarl file, DE europarl file, number of sentences to use from the europarl files, EN reuters directory, in case of choosing 'union' or 'intersection'

    'union' := the vocabulary is words that appear in either europarl or reuters datasets
    'intersection' := the vocabulary is words that appear in both europarl and reuters datasets
    'europarl' := the vocabulary is words that appear only in the europarl dataset
    Note: these refer only to the English vocabulary, German vocabulary is always 'europarl'

    'bow' := bag of words representation
    'LSA' := latent semantic analysis representation
    
    returns word_to_indexEN, word_to_indexDE, EN, DE '''
    return {'union': get_counts_union, 'intersection': get_counts_intersection, 'europarl': get_counts_europarl}[vocab](data, method, config.ALIGNED_EN_FILE, config.ALIGNED_DE_FILE, nr_of_sentences, reuters, rev)

def update_dicts(sentence, freqs, total_counts, counts, method, add=True, divide=False):
    if add and divide:
        raise NotImplementedError()
    for i in xrange(len(sentence)):
        if method == 'LSA':
            if config.SYMMETRIC_PMI:
                r = xrange(max(0,i-config.WINDOW_SIZE),min(len(sentence),i+config.WINDOW_SIZE+1))
            else:
                r = xrange(i, min(len(sentence), i+config.WINDOW_SIZE+1))

            if config.COOC_ONLY_ONE:
                words = []
            if divide:
                c = 0.
                for j in r:
                    if (sentence[i], sentence[j]) in freqs and sentence[i] != sentence[j] and counts[sentence[j]] > config.COOC_THRESH:
                        c += 1.
                if c == 0.:
                    continue
###
            tfreqs = 0.
###
            for j in r:
                if (sentence[i], sentence[j]) in freqs:
                    if config.COOC_ONLY_ONE:
                        if sentence[j] not in words:
                            words.append(sentence[j])
                        else:
                            continue
                    if i != j and divide and sentence[i] == sentence[j]:
                        continue
                    if divide and sentence[i] != sentence[j]:
                        if counts[sentence[j]] > config.COOC_THRESH:
                            freqs[sentence[i], sentence[j]] += 1./c
                        else:
                            continue
###
                        tfreqs += 1./c
###
                    else:
                        freqs[sentence[i], sentence[j]] += 1
                elif add:
                    if config.COOC_ONLY_ONE:
                        words.append(sentence[j])
                    freqs[sentence[i], sentence[j]] = 1
        if sentence[i] in counts and not divide:
            counts[sentence[i]] += 1
        elif add:
            counts[sentence[i]] = 1
        total_counts += 1

    return total_counts
    

def get_counts_europarl(data, method, ENeuroparl, DEeuroparl, nr_sentences, ENreuters, rev):
    ENfreqs = {} # co-occurencies
    DEfreqs = {}
    total_countsEN = 0
    total_countsDE = 0
    ENcounts = {} # nr of times a word appears
    DEcounts = {}
    with codecs.open(ENeuroparl, 'r','utf-8') as fileEN:
        with codecs.open(DEeuroparl, 'r','utf-8') as fileDE:
            n = 0
            for lineEN, lineDE in zip(fileEN, fileDE):
                if lineEN in ['\n','\r\n'] or lineDE in ['\n','\r\n']: # ignore empty lines
                    continue
                sentenceEN = lineEN.lower().split()
                total_countsEN = update_dicts(sentenceEN, ENfreqs, total_countsEN, ENcounts, method)
                sentenceDE = lineDE.lower().split()
                total_countsDE = update_dicts(sentenceDE, DEfreqs, total_countsDE, DEcounts, method)

                if nr_sentences != 0 and n > nr_sentences:
                    break
                n += 1

    if method == 'bow':
        return ENcounts, DEcounts
    if method == 'LSA':
        return ENcounts, DEcounts, ENfreqs, DEfreqs, total_countsEN, total_countsDE
    raise NotImplementedError(str(method) + ' is unknown method')

def get_counts_union(data, method, ENeuroparl, DEeuroparl, nr_of_sentences, ENreuters, rev):
    # vocabulary from europarl
    if method == 'bow':
        ENcounts, DEcounts = get_counts_europarl(data, method, ENeuroparl, DEeuroparl, nr_of_sentences, ENreuters, rev)
        ENfreqs = DEfreqs = {} # these aren't needed
        total_countsEN = total_countsDE = 0 # these aren't needed
    elif method == 'LSA':
        ENcounts, DEcounts, ENfreqs, DEfreqs, total_countsEN, total_countsDE = get_counts_europarl(data, method, ENeuroparl, DEeuroparl, nr_of_sentences, ENreuters, rev)
    else:
        raise NotImplementedError(str(method) + ' is unknown method')

    # vocabulary from reuters
    for dir in ['C','E','G','M']: 
        for filename in sorted(os.listdir(config.REUTERS_TRAIN_DIR + ENreuters + os.sep + dir)):
            with codecs.open(config.REUTERS_TRAIN_DIR + ENreuters + os.sep + dir + os.sep + filename, 'r', 'utf-8') as f:
                for line in f:
                    sentence = line.lower().split()
                    if rev:
                        total_countsDE = update_dicts(sentence, DEfreqs, total_countsDE, DEcounts, method)
                    else:
                        total_countsEN = update_dicts(sentence, ENfreqs, total_countsEN, ENcounts, method)

    if method == 'bow':
        return ENcounts, DEcounts
    if method == 'LSA':
        return ENcounts, DEcounts, ENfreqs, DEfreqs, total_countsEN, total_countsDE

def get_counts_reuters(data, DEcounts, DEfreqs, total_countsDE, DEreuters):
    # vocabulary from reuters
    for dir in ['C','E','G','M']: 
        for filename in sorted(os.listdir(config.REUTERS_TRAIN_DIR + DEreuters + os.sep + dir)):
            with codecs.open(config.REUTERS_TRAIN_DIR + DEreuters + os.sep + dir + os.sep + filename, 'r', 'utf-8') as f:
                for line in f:
                    sentence = line.lower().split()
                    total_countsDE = update_dicts(sentence, DEfreqs, total_countsDE, DEcounts, 'LSA', False)

    return DEcounts, DEfreqs, total_countsDE

def get_divide_counts(ENcounts, DEcounts, ENfreqs, DEfreqs, total_countsEN, total_countsDE, DEreuters, ENeuroparl, DEeuroparl, nr_sentences, ENreuters, rev):
    for keywords, val in ENfreqs.items():
        ENfreqs[keywords] = 0.
    for keywords, val in DEfreqs.items():
        DEfreqs[keywords] = 0.
    total_countsEN = 0
    total_countsDE = 0

    # europarl
    with codecs.open(ENeuroparl, 'r','utf-8') as fileEN:
        with codecs.open(DEeuroparl, 'r','utf-8') as fileDE:
            n = 0
            for lineEN, lineDE in zip(fileEN, fileDE):
                if lineEN in ['\n','\r\n'] or lineDE in ['\n','\r\n']: # ignore empty lines
                    continue
                sentenceEN = lineEN.lower().split()
                total_countsEN = update_dicts(sentenceEN, ENfreqs, total_countsEN, ENcounts, "LSA", False, True)
                sentenceDE = lineDE.lower().split()
                total_countsDE = update_dicts(sentenceDE, DEfreqs, total_countsDE, DEcounts, "LSA", False, True)

                if nr_sentences != 0 and n > nr_sentences:
                    break
                n += 1
    # reuters source
    for dir in ['C','E','G','M']: 
        for filename in sorted(os.listdir(config.REUTERS_TRAIN_DIR + ENreuters + os.sep + dir)):
            with codecs.open(config.REUTERS_TRAIN_DIR + ENreuters + os.sep + dir + os.sep + filename, 'r', 'utf-8') as f:
                for line in f:
                    sentence = line.lower().split()
                    if rev:
                        total_countsDE = update_dicts(sentence, DEfreqs, total_countsDE, DEcounts, "LSA", False, True)
                    else:
                        total_countsEN = update_dicts(sentence, ENfreqs, total_countsEN, ENcounts, "LSA", False, True)
    # reuters target
    for dir in ['C','E','G','M']: 
        for filename in sorted(os.listdir(config.REUTERS_TRAIN_DIR + DEreuters + os.sep + dir)):
            with codecs.open(config.REUTERS_TRAIN_DIR + DEreuters + os.sep + dir + os.sep + filename, 'r', 'utf-8') as f:
                for line in f:
                    sentence = line.lower().split()
                    if rev:
                        total_countsEN = update_dicts(sentence, ENfreqs, total_countsEN, ENcounts, 'LSA', False, True)
                    else:
                        total_countsDE = update_dicts(sentence, DEfreqs, total_countsDE, DEcounts, 'LSA', False, True)

    return ENcounts, DEcounts, ENfreqs, DEfreqs, total_countsEN, total_countsDE

def get_counts_intersection(data, method, ENeuroparl, DEeuroparl, nr_sentences, ENreuters, counts = False):
    raise NotImplementedError()
