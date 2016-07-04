import numpy as np
import sys
import pdb

def read_multilabel_dataset(filepath, num_labels=-1, num_features=-1, \
                            has_label_probabilities=False, sparse=False, add_bias=False):
    f = open(filepath)
    all_labels = []
    all_label_probabilities = []
    all_features = []
    for line in f:
        line = line.rstrip('\n')
        fields = line.split()
        #if fields[0] == '70': pdb.set_trace()
        if has_label_probabilities:
            labels, label_probabilities = zip(*[l.split(':') for l in fields[0].split(',')])
            labels = [int(l) for l in labels]
            label_probabilities = [float(val) for val in label_probabilities]
        else:
            labels = [int(l) for l in fields[0].split(',')]
        features = {}
        if add_bias:
            features[1] = 1.
        for field in fields[1:]:
            name_value = field.split(':')
            assert len(name_value) == 2, pdb.set_trace()
            fid = int(name_value[0])
            fval = float(name_value[1])
            assert fid > 0, pdb.set_trace() # 0 is reserved for UNK.
            if add_bias:
                fid += 1 # 1 is reserved for bias feature.
            assert fid not in features, pdb.set_trace()
            if num_features >= 0 and fid >= num_features:
                fid = 0 # UNK.
            features[fid] = fval
        #pdb.set_trace()
        all_labels.append(labels)
        if has_label_probabilities:
            all_label_probabilities.append(label_probabilities)
        all_features.append(features)
    f.close()

    label_set = set.union(*[set(labels) for labels in all_labels])
    #label_set = set()
    #for labels in all_labels:
    #    label_set = label_set.union(set(labels))
    feature_set = set.union(*[set(features.keys()) for features in all_features])
    print 'Num examples: %d' % len(all_labels)
    print 'Num labels: %d, max label: %d, min label: %d' % (len(label_set), max(label_set), min(label_set))
    print 'Num features: %d, max feature: %d, min feature: %d' % (len(feature_set), max(feature_set), min(feature_set))
    #pdb.set_trace()

    num_examples = len(all_labels)
    if num_features < 0:
        num_features = 1+max(feature_set)
    if num_labels < 0:
        num_labels = 1+max(label_set)
    else:
        # Make sure there are no UNK labels in the test set.
        assert num_labels == 1+max(label_set), pdb.set_trace()

    Y = np.zeros((num_examples, num_labels), dtype=float)
    if has_label_probabilities:
        for i, (labels, label_probabilities) in \
            enumerate(zip(all_labels, all_label_probabilities)):
            for label, val in zip(labels, label_probabilities):
                Y[i, label] = val
    else:
        for i, labels in enumerate(all_labels):
            for label in labels:
                Y[i, label] = 1
    if sparse:
        X_sparse = all_features
        return X_sparse, Y, num_features
    else:
        X = np.zeros((num_examples, num_features))
        for i, features in enumerate(all_features):
            for fid, fval in features.iteritems():
                assert fid < num_features, pdb.set_trace()
                X[i, fid] = fval
        return X, Y, num_features


if __name__ == "__main__":
    read_multilabel_dataset(sys.argv[1])
