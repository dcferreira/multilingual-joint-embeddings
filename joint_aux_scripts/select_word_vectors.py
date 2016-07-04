import sys
import pdb

word_vectors_filepath = sys.argv[1]
data_filepath = sys.argv[2]
skip_first_line = False #True

f = open(word_vectors_filepath)
# Skip first line.
if skip_first_line:
    print >> sys.stderr, 'Skipping first line...'
    f.readline()
index = 0
word_indices = {}
for line in f:
    line = line.rstrip('\n')
    fields = line.split(' ')
    word = fields[0]
    if word in word_indices:
        print >> sys.stderr, 'Repeated word: %s' % word
    #assert word not in word_indices, pdb.set_trace()
    word_indices[word] = index
    index += 1
f.close()

print >> sys.stderr, 'Number of word vectors: %d' % index
active_indices = [False] * index

f = open(data_filepath)
data_words = set()
for line in f:
    line = line.rstrip('\n')
    fields = line.split('\t')
    assert len(fields) == 2, pdb.set_trace()
    assert fields[0][-3:] == '_en', pdb.set_trace()
    word = fields[0][:-3] # To remove '_en'
    if word not in data_words:
        data_words.add(word)
f.close()

print >> sys.stderr, 'Number of data words: %d' % len(data_words)

num_unknown_words = 0
for word in data_words:
    if word in word_indices:
        index = word_indices[word]
        active_indices[index] = True
    else:
        num_unknown_words += 1

print >> sys.stderr, 'Number of unknown data words: %d' % num_unknown_words

# Dump the new word vector file.
f = open(word_vectors_filepath)
if skip_first_line:
    print >> sys.stderr, 'Skipping first line...'
    print f.readline().rstrip('\n')
else:
    print 'Selection from %s' % word_vectors_filepath
index = 0
for line in f:
    line = line.rstrip('\n')
    if active_indices[index]:
        print line
    index += 1
f.close()
