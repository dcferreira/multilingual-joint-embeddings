import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from multilabel_dataset_reader import read_multilabel_dataset
import time
import sys
import pdb

def compute_support(probs):
    ind = probs.nonzero()[0]
    supp =  np.zeros_like(probs)
    supp[ind] = 1.
    return supp

def project_onto_simplex(a, radius=1.0):
    '''Project point a to the probability simplex.
    Returns the projected point x and the residual value.'''
    x0 = a.copy()
    d = len(x0);
    ind_sort = np.argsort(-x0)
    y0 = x0[ind_sort]
    ycum = np.cumsum(y0)
    val = 1.0/np.arange(1,d+1) * (ycum - radius)
    ind = np.nonzero(y0 > val)[0]
    rho = ind[-1]
    tau = val[rho]
    y = y0 - tau
    ind = np.nonzero(y < 0)
    y[ind] = 0
    x = x0.copy()
    x[ind_sort] = y

    return x, tau, .5*np.dot(x-a, x-a)

def evaluate_and_compute_gradient(weights_flattened, *args):
    X_train = args[0]
    Y_train = args[1]
    loss_function = args[2]
    regularization_constant = args[3]
    add_bias = args[4]
    has_label_probabilities = args[5]

    num_documents_train = len(X_train)

    weights = weights_flattened.reshape(num_words, num_classes)
    gradient = np.zeros_like(weights)

    loss = 0.
    for i in xrange(num_documents_train):
        if has_label_probabilities:
            y = Y_train[i,:].copy()
        else:
            y = Y_train[i,:].copy() / sum(Y_train[i,:])

        x = X_train[i]
        scores = np.zeros(num_classes)
        for fid, fval in x.iteritems():
            scores += fval * weights[fid, :]

        gold_labels = compute_support(y)

        predicted_labels_eval = []
        if loss_function == 'sparsemax':
            probs, tau, _ =  project_onto_simplex(scores)
            predicted_labels = compute_support(probs)
            loss_t = -scores.dot(y) + \
                     .5*(scores**2 - tau**2).dot(predicted_labels) + \
                     .5 * sum(y**2)
            loss += loss_t
            assert loss_t > -1e-9 #, pdb.set_trace()
            delta = -y + probs
            for fid, fval in x.iteritems():
                gradient[fid] += fval * delta
        elif loss_function == 'softmax':
            probs = np.exp(scores) / np.sum(np.exp(scores))
            loss_t = -scores.dot(y) + np.log(np.sum(np.exp(scores)))
            loss += loss_t
            assert loss_t > -1e-9 #, pdb.set_trace()
            delta = -y + probs
            for fid, fval in x.iteritems():
                gradient[fid] += fval * delta
        elif loss_function == 'logistic':
            probs = 1. / (1. + np.exp(-scores))
            log_probs = -np.log(1. + np.exp(-scores))
            log_neg_probs = -np.log(1. + np.exp(scores))
            loss_t = \
                -log_probs.dot(gold_labels) - log_neg_probs.dot(1. - gold_labels)
            #loss_t = \
            #    -np.log(probs).dot(gold_labels) - np.log(1. - probs).dot(1. - gold_labels)
            loss += loss_t
            if not loss_t > -1e-6:
                print loss_t
            assert loss_t > -1e-6 #, pdb.set_trace()
            delta = -gold_labels + probs
            for fid, fval in x.iteritems():
                gradient[fid] += fval * delta

    gradient /= num_documents_train
    loss /= num_documents_train
    if add_bias:
        gradient[2:,:] += regularization_constant * weights[2:,:]
        reg = 0.5 * regularization_constant * np.linalg.norm(weights[2:,:].flatten())**2
    else:
        gradient += regularization_constant * weights
        reg = 0.5 * regularization_constant * np.linalg.norm(weights.flatten())**2

    objective_value = loss + reg

    print 'Loss: %f, Reg: %f, Loss+Reg: %f' % (loss, reg, objective_value)
    return objective_value, gradient.flatten()



def classify_dataset(filepath, weights, classifier_type, \
                     hyperparameter_name, \
                     hyperparameter_values,
                     start_position=-1, \
                     end_position=-1, \
                     has_label_probabilities=False, \
                     add_bias=False, \
                     normalize=False, \
                     x_av=None, \
                     x_std=None):
    num_settings = len(hyperparameter_values)

    rank_acc = np.zeros(num_settings)
    matched_labels = np.zeros(num_settings)
    union_labels = np.zeros(num_settings)
    num_correct = np.zeros(num_settings)
    num_total = np.zeros(num_settings)
    num_predicted_labels = np.zeros(num_settings)
    num_gold_labels = np.zeros(num_settings)
    squared_loss_dev = 0.
    JS_loss_dev = 0.

    num_features = weights.shape[0]
    num_labels = weights.shape[1]

    num_documents = 0

    num_matched_by_label = np.zeros((num_settings, num_labels))
    num_predicted_by_label = np.zeros((num_settings, num_labels))
    num_gold_by_label = np.zeros((num_settings, num_labels))

    position = 0
    f = open(filepath)
    for line in f:
        line = line.rstrip('\n')
        fields = line.split()
        if has_label_probabilities:
            labels, label_probabilities = zip(*[l.split(':') for l in fields[0].split(',')])
            labels = [int(l) for l in labels]
            label_probabilities = [float(val) for val in label_probabilities]
        else:
            labels = [int(l) for l in fields[0].split(',')]
        features = {}
        if start_position >= 0 and end_position >= 0 and \
           (start_position > position or end_position <= position):
            position += 1
            continue
        position += 1
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
            if normalize:
                if x_std[fid] != 0:
                    features[fid] -= x_av[fid]
                    features[fid] /= x_std[fid]

        # Now classify this instance.
        x = features
        y = np.zeros(num_labels, dtype=float)
        if has_label_probabilities:
            for label, val in zip(labels, label_probabilities):
                y[label] = val
        else:
            for label in labels:
                y[label] = 1.
            y /= sum(y)

        scores = np.zeros(num_labels)
        for fid, fval in x.iteritems():
            scores += fval * weights[fid, :]

        gold_labels = compute_support(y)
        predicted_labels_eval = []

        if classifier_type == 'sparsemax':
            probs, _, _ =  project_onto_simplex(scores)
            for sparsemax_scale in hyperparameter_values:
                scaled_probs, _, _ =  project_onto_simplex(sparsemax_scale * scores)
                predicted_labels = compute_support(scaled_probs)
                predicted_labels_eval.append(predicted_labels)
        elif classifier_type == 'softmax':
            probs = np.exp(scores) / np.sum(np.exp(scores))
            for probability_threshold in hyperparameter_values:
                predicted_labels = (probs > probability_threshold).astype(float)
                predicted_labels_eval.append(predicted_labels)
        elif classifier_type == 'logistic':
            probs = 1. / (1. + np.exp(-scores))
            for probability_threshold in hyperparameter_values:
                predicted_labels = (probs > probability_threshold).astype(float)
                predicted_labels_eval.append(predicted_labels)
        else:
            raise NotImplementedError

        squared_loss_dev += sum((probs - y)**2)
        p_av = (probs + y) / 2.
        ind_probs = np.nonzero(probs)[0]
        ind_y = np.nonzero(y)[0]
        if classifier_type != 'logistic':
            JS_loss_dev += .5*np.dot(probs[ind_probs], np.log(probs[ind_probs]/p_av[ind_probs])) + \
                                .5*np.dot(y[ind_y], np.log(y[ind_y]/p_av[ind_y]))

        num_gold = sum(gold_labels)
        for k in xrange(len(hyperparameter_values)):
            predicted_labels = predicted_labels_eval[k]
            num_predicted = sum(predicted_labels)
            num_matched = gold_labels.dot(predicted_labels)
            num_union = sum(compute_support(gold_labels + predicted_labels))
            assert num_union == num_predicted + num_gold - num_matched

            for l in xrange(num_labels):
                if predicted_labels[l] == 1:
                    num_predicted_by_label[k, l] += 1.
                    if gold_labels[l] == 1:
                        num_matched_by_label[k, l] += 1.
                if gold_labels[l] == 1:
                    num_gold_by_label[k, l] += 1.

            if num_labels != num_gold:
                rank_acc[k] += float(num_matched * (num_labels - num_union)) / float(num_gold * (num_labels - num_gold))
            else:
                rank_acc[k] += 0.
            matched_labels[k] += num_matched
            union_labels[k] += num_union
            num_gold_labels[k] += num_gold
            num_predicted_labels[k] += num_predicted

            num_correct[k] += sum((gold_labels == predicted_labels).astype(float))
            num_total[k] += len(gold_labels)

        num_documents += 1

    f.close()

    JS_loss_dev /= (num_documents*num_labels)
    squared_loss_dev /= (num_documents*num_labels)

    print 'Number of documents in %s: %d, sq loss: %f, JS loss: %f' % \
        (filepath, num_documents, squared_loss_dev, JS_loss_dev)

    acc_dev = matched_labels / union_labels
    hamming_dev = num_correct / num_total
    P_dev = matched_labels / num_predicted_labels
    R_dev = matched_labels / num_gold_labels
    F1_dev = 2*P_dev*R_dev / (P_dev + R_dev)

    Pl_dev = num_matched_by_label / num_predicted_by_label
    Rl_dev = num_matched_by_label / num_gold_by_label
    F1l_dev = 2*Pl_dev*Rl_dev / (Pl_dev + Rl_dev)

    Pl_dev = np.nan_to_num(Pl_dev) # Replace nans with zeros.
    Rl_dev = np.nan_to_num(Rl_dev) # Replace nans with zeros.
    F1l_dev = np.nan_to_num(F1l_dev) # Replace nans with zeros.

    rank_acc /= float(num_documents)

    print_all_labels = False

    if False:
        for k in xrange(len(hyperparameter_values)):
        
            macro_P_dev = np.mean(Pl_dev[k, :])
            macro_R_dev = np.mean(Rl_dev[k, :])
            macro_F1_dev = 2*macro_P_dev*macro_R_dev / (macro_P_dev + macro_R_dev)
            #macro_F1_dev_wrong = np.mean(F1l_dev[k, :])
            print '%s: %f, acc_dev: %f, hamming_dev: %f, P_dev: %f, R_dev: %f, F1_dev: %f, macro_P_dev: %f, macro_R_dev: %f, macro_F1_dev: %f, rank_acc: %f' % \
                (hyperparameter_name, hyperparameter_values[k], \
                 acc_dev[k], hamming_dev[k], P_dev[k], R_dev[k], F1_dev[k], macro_P_dev, macro_R_dev, macro_F1_dev, rank_acc[k])

            if print_all_labels:
                for l in xrange(num_labels): 
                    print '  LABEL %d, %s: %f,  P_dev: %f, R_dev: %f, F1_dev: %f' % \
                        (l, hyperparameter_name, hyperparameter_values[k], \
                         Pl_dev[k, l], Rl_dev[k, l], F1l_dev[k, l])

    print_accuracies(hyperparameter_name, hyperparameter_values, \
                     acc_dev, hamming_dev, P_dev, R_dev, F1_dev, Pl_dev, Rl_dev, rank_acc)

    return squared_loss_dev, JS_loss_dev, acc_dev, hamming_dev, P_dev, R_dev, F1_dev, Pl_dev, Rl_dev, rank_acc


def print_accuracies(hyperparameter_name, hyperparameter_values, \
                     acc_dev, hamming_dev, P_dev, R_dev, F1_dev, Pl_dev, Rl_dev, rank_acc):
    print_all_labels = False

    for k in xrange(len(hyperparameter_values)):
        
        macro_P_dev = np.mean(Pl_dev[k, :])
        macro_R_dev = np.mean(Rl_dev[k, :])
        macro_F1_dev = 2*macro_P_dev*macro_R_dev / (macro_P_dev + macro_R_dev)
        #macro_F1_dev_wrong = np.mean(F1l_dev[k, :])
        print '%s: %f, acc_dev: %f, hamming_dev: %f, P_dev: %f, R_dev: %f, F1_dev: %f, macro_P_dev: %f, macro_R_dev: %f, macro_F1_dev: %f, rank_acc: %f' % \
            (hyperparameter_name, hyperparameter_values[k], \
             acc_dev[k], hamming_dev[k], P_dev[k], R_dev[k], F1_dev[k], macro_P_dev, macro_R_dev, macro_F1_dev, rank_acc[k])

        if print_all_labels:
            for l in xrange(num_labels): 
                print '  LABEL %d, %s: %f,  P_dev: %f, R_dev: %f, F1_dev: %f' % \
                    (l, hyperparameter_name, hyperparameter_values[k], \
                     Pl_dev[k, l], Rl_dev[k, l], F1l_dev[k, l])


def print_accuracies_means_and_deviations(hyperparameter_name, hyperparameter_values, \
                                          acc, hamming, P, R, F1, Pl, Rl, rank_acc):
    F1_av_all = np.zeros(len(hyperparameter_values))
    macro_F1_av_all = np.zeros(len(hyperparameter_values))

    for k in xrange(len(hyperparameter_values)):

        acc_av = np.mean(np.array([val[k] for val in acc]))
        acc_std = np.std(np.array([val[k] for val in acc]))
        hamming_av = np.mean(np.array([val[k] for val in hamming]))
        hamming_std = np.std(np.array([val[k] for val in hamming]))
        P_av = np.mean(np.array([val[k] for val in P]))
        P_std = np.std(np.array([val[k] for val in P]))
        R_av = np.mean(np.array([val[k] for val in R]))
        R_std = np.std(np.array([val[k] for val in R]))
        F1_av = np.mean(np.array([val[k] for val in F1]))
        F1_std = np.std(np.array([val[k] for val in F1]))
        macro_P = []
        macro_R = []
        macro_F1 = []
        for t in xrange(len(Pl)):
            macro_P.append(np.mean(Pl[t][k, :]))
            macro_R.append(np.mean(Rl[t][k, :]))
            macro_F1.append(2*macro_P[-1]*macro_R[-1] / (macro_P[-1] + macro_R[-1]))
        macro_P_av = np.mean(np.array([val for val in macro_P]))
        macro_P_std = np.std(np.array([val for val in macro_P]))
        macro_R_av = np.mean(np.array([val for val in macro_R]))
        macro_R_std = np.std(np.array([val for val in macro_R]))
        macro_F1_av = np.mean(np.array([val for val in macro_F1]))
        macro_F1_std = np.std(np.array([val for val in macro_F1]))
        rank_acc_av = np.mean(np.array([val[k] for val in rank_acc]))
        rank_acc_std = np.std(np.array([val[k] for val in rank_acc]))

        F1_av_all[k] = F1_av
        macro_F1_av_all[k] = macro_F1_av

        print '%s: %f, acc_dev: %f +- %f, hamming_dev: %f +- %f, P_dev: %f +- %f, R_dev: %f +- %f, F1_dev: %f +- %f, macro_P_dev: %f +- %f, macro_R_dev: %f +- %f, macro_F1_dev: %f +- %f, rank_acc: %f +- %f' % \
            (hyperparameter_name, hyperparameter_values[k], \
             acc_av, acc_std, hamming_av, hamming_std, P_av, P_std, R_av, R_std, F1_av, F1_std, macro_P_av, macro_P_std, macro_R_av, macro_R_std, macro_F1_av, macro_F1_std, rank_acc_av, rank_acc_std)

    print
    k = np.argmax(np.nan_to_num(F1_av_all))
    print 'Best %s optimized for micro-F1: %f (micro-F1 = %f).' % (hyperparameter_name, hyperparameter_values[k], F1_av_all[k])
    k = np.argmax(np.nan_to_num(macro_F1_av_all))
    print 'Best %s optimized for macro-F1: %f (macro-F1 = %f).' % (hyperparameter_name, hyperparameter_values[k], macro_F1_av_all[k])



###########################

loss_function = sys.argv[1] #'softmax' #'logistic' # 'sparsemax'
num_epochs = int(sys.argv[2]) #20
regularization_constant = float(sys.argv[3])

filepath_train = sys.argv[4]
filepath_test = sys.argv[5]
add_bias = True # False
normalize = bool(int(sys.argv[6]))
has_label_probabilities = False #True

num_jackknife_partitions = int(sys.argv[7]) # 1 means no jackknifing.

X_train, Y_train, num_features = \
    read_multilabel_dataset(filepath_train, \
                            add_bias=add_bias, \
                            has_label_probabilities=has_label_probabilities, \
                            sparse=True)
if normalize:
    X = np.zeros((len(X_train), num_features))
    for i, features in enumerate(X_train):
        for fid, fval in features.iteritems():
            assert fid < num_features, pdb.set_trace()
            X[i, fid] = fval
    x_av = X.mean(axis=0)
    x_std = X.std(axis=0)
    #x_std = np.sqrt(((X-x_av)*(X-x_av)).sum(axis=0) / X.shape[0])
    for i, features in enumerate(X_train):
        for fid, fval in features.iteritems():
            if x_std[fid] != 0:
                features[fid] -= x_av[fid]
                features[fid] /= x_std[fid]
else:
    x_av = None
    x_std = None
        
num_labels = Y_train.shape[1]

num_words = num_features
num_classes = num_labels
num_documents_train = len(X_train)

print 'Label frequencies:', Y_train.sum(axis=0) 

sparsemax_scales = np.arange(1, 5.5, .5)
softmax_thresholds = np.arange(1., 11., 1.) / num_labels #np.arange(.02, 0.22, .02)
logistic_thresholds = np.arange(.05, 0.55, .05)

if loss_function == 'softmax':
    hyperparameter_name = 'softmax_thres'
    hyperparameter_values = softmax_thresholds
elif loss_function == 'sparsemax':
    hyperparameter_name = 'sparsemax_scale'
    hyperparameter_values = sparsemax_scales
elif loss_function == 'logistic':
    hyperparameter_name = 'logistic_thres'
    hyperparameter_values = logistic_thresholds
else:
    raise NotImplementedError

if num_jackknife_partitions <= 1:

    weights = np.zeros((num_words, num_classes))
    weights_flattened = weights.flatten()

    # Optimize with L-BFGS.
    weights_flattened, value, d = \
      opt.fmin_l_bfgs_b(evaluate_and_compute_gradient,
                        x0=weights_flattened,
                        args=(X_train, Y_train, loss_function, regularization_constant, add_bias, has_label_probabilities),
                        m=10,
                        factr=100,
                        pgtol=1e-08,
                        epsilon=1e-12,
                        approx_grad=False,
                        disp=True,
                        maxfun=1000,
                        maxiter=num_epochs) #1000)
                            
    weights = weights_flattened.reshape(num_words, num_classes)

    if d['warnflag'] != 0:
        print 'Not converged', d

    print 'Running on the test set...'
    tic = time.time()
    classify_dataset(filepath_test, weights, loss_function, \
                     hyperparameter_name, \
                     hyperparameter_values, \
                     has_label_probabilities=has_label_probabilities, \
                     add_bias=add_bias, \
                     normalize=normalize, \
                     x_av=x_av, \
                     x_std=x_std)
    elapsed_time = time.time() - tic
    print 'Time to test: %f' % elapsed_time

else:
    num_examples = len(X_train)
    partition_size = int(np.ceil(float(num_examples) / num_jackknife_partitions))
    start_positions = np.arange(0, num_examples, partition_size)
    end_positions = start_positions + partition_size
    end_positions[-1] = num_examples
    squared_loss = []
    JS_loss = []
    acc = []
    hamming = []
    P = []
    R = []
    F1 = []
    Pl = []
    Rl = []
    rank_acc = []
    for t in xrange(num_jackknife_partitions):
        print
        print '-----------------------------------------'
        print 'Jackknife partition %d...' % (t+1)
        print '-----------------------------------------'
        print

        X_train_t = X_train[:start_positions[t]] + X_train[end_positions[t]:]
        Y_train_t = np.concatenate([Y_train[:start_positions[t], :], Y_train[end_positions[t]:, :]], axis=0) 

        weights = np.zeros((num_words, num_classes))
        weights_flattened = weights.flatten()

        # Optimize with L-BFGS.
        weights_flattened, value, d = \
          opt.fmin_l_bfgs_b(evaluate_and_compute_gradient,
                            x0=weights_flattened,
                            args=(X_train_t, Y_train_t, loss_function, regularization_constant, add_bias, has_label_probabilities),
                            m=10,
                            factr=100,
                            pgtol=1e-08,
                            epsilon=1e-12,
                            approx_grad=False,
                            disp=False,
                            maxfun=1000,
                            maxiter=num_epochs) #1000)
                            
        weights = weights_flattened.reshape(num_words, num_classes)

        if d['warnflag'] != 0:
            print 'Not converged', d

        print 'Running on the dev set...'
        tic = time.time()
        squared_loss_dev, JS_loss_dev, acc_dev, hamming_dev, P_dev, R_dev, F1_dev, Pl_dev, Rl_dev, rank_acc_dev = \
            classify_dataset(filepath_train, weights, loss_function, \
                             hyperparameter_name, \
                             hyperparameter_values, \
                             start_position=start_positions[t], \
                             end_position=end_positions[t], \
                             has_label_probabilities=has_label_probabilities, \
                             add_bias=add_bias, \
                             normalize=normalize, \
                             x_av=x_av, \
                             x_std=x_std)
        squared_loss.append(squared_loss_dev)
        JS_loss.append(JS_loss_dev)
        acc.append(acc_dev)
        hamming.append(hamming_dev)
        P.append(P_dev)
        R.append(R_dev)
        F1.append(F1_dev)
        Pl.append(Pl_dev)
        Rl.append(Rl_dev)
        rank_acc.append(rank_acc_dev)

        elapsed_time = time.time() - tic
        print 'Time to test: %f' % elapsed_time

    print 'Sq loss: %f +- %f, JS loss: %f +- %f' % \
        (np.mean(np.array(squared_loss)), \
         np.std(np.array(squared_loss)), \
         np.mean(np.array(JS_loss)), \
         np.std(np.array(JS_loss)))
    print_accuracies_means_and_deviations(hyperparameter_name, hyperparameter_values, \
                                          acc, hamming, P, R, F1, Pl, Rl, rank_acc)


