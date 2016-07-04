import sys
import numpy as np
import time
from scipy.sparse import csr_matrix, dok_matrix
import pdb

class EmbeddingTrainer(object):
    def __init__(self):
        pass

    def load_vocabulary(self, vocab_filepath):
        V = {}
        f = open(vocab_filepath)
        for line in f:
            word = line.rstrip('\n').split('\t')[0]
            assert word not in V, pdb.set_trace()
            wid = len(V)
            V[word] = wid
        f.close()
        return V

    def dump_embeddings(self, P, vocab, embeddings_filepath):
        f = open(embeddings_filepath, 'w')
        for word in vocab:
            wid = vocab[word]
            embedding = list(P[wid, :])
            f.write(word + ' ' + ' '.join([str(val) for val in embedding]) + '\n')
        f.close()

    def load_embeddings(self, embeddings_filepath, vocab, dimension):
        skip_first_line = False
        f = open(embeddings_filepath)
        # Skip first line.
        if skip_first_line:
            print >> sys.stderr, 'Skipping first line...'
            f.readline()
        E = np.zeros((len(vocab), dimension))
        num_covered = 0
        for line in f:
            line = line.rstrip('\n')
            fields = line.split(' ')
            word = fields[0] + '_en'
            #pdb.set_trace()
            if word in vocab:
                num_covered += 1
                embedding = np.array([float(val) for val in fields[1:]])
                assert embedding.shape[0] == dimension, pdb.set_trace()
                wid = vocab[word]
                E[wid, :] = embedding
        print >> sys.stderr, 'Covered %d out of %d words.' % (num_covered,
                                                              len(vocab))
        return E

    def load_data(self, filepath, V):
        f = open(filepath)
        S = []
        for line in f:
            s = {}
            words = line.rstrip('\n').split(' ')
            for word in words:
                if word in V:
                    wid = V[word]
                    if wid in s:
                        s[wid] += 1
                    else:
                        s[wid] = 1
            S.append(s)
        f.close()

        # Transform to dok and to csr.
        Sdok = dok_matrix((len(S), len(V)), dtype=int)
        for n, s in enumerate(S):
            for wid in s:
                Sdok[n, wid] = s[wid]
        S = Sdok.tocsr()
        return S

    def compute_residuals(self, S, T, P, Q):
        self.residuals = S.dot(P) - T.dot(Q)

    def compute_gradient(self, S, T, P0, mu, mu_s, mu_t, P, Q):
        # DP = mu*S*(S'*P - T'*Q)/N + (1/V+mu_s) P - P0/V
        # DQ = -mu*T*(S'*P - T'*Q)/N + mu_t Q
        # We define residuals = S'*P - T'*Q (N-by-K matrix).
        V_s = float(P.shape[0])
        if self.regularization_type == 'l2':
            self.DP = mu/N * S.T.dot(self.residuals) + (1./V_s + mu_s) * P - P0/V_s
            self.DQ = -mu/N * T.T.dot(self.residuals) + mu_t * Q
        else:
            # l1.
            self.DP = mu/N * S.T.dot(np.sign(self.residuals)) + (1./V_s + mu_s) * P - P0/V_s
            self.DQ = -mu/N * T.T.dot(np.sign(self.residuals)) + mu_t * Q


if __name__ == "__main__":
    source_filepath = sys.argv[1]
    target_filepath = sys.argv[2]
    source_vocab_filepath = sys.argv[3]
    target_vocab_filepath = sys.argv[4]
    embeddings_filepath = sys.argv[5]
    regularization_type = sys.argv[6] # 'l1' or 'l2'.
    mu = float(sys.argv[7]) #0.1
    mu_s = 0.
    mu_t = 0.
    num_epochs = int(sys.argv[8]) #50
    source_embeddings_filepath = sys.argv[9] # P.txt
    target_embeddings_filepath = sys.argv[10] # Q.txt
    eta = 1.
    epsilon = 1e-12
    dimension = 300

    trainer = EmbeddingTrainer()
    trainer.regularization_type = regularization_type

    print >> sys.stderr, "Loading source vocabulary..."
    source_vocab = trainer.load_vocabulary(source_vocab_filepath)
    print >> sys.stderr, "Loading target vocabulary..."
    target_vocab = trainer.load_vocabulary(target_vocab_filepath)
    print >> sys.stderr, "Loading source embeddings..."
    P0 = trainer.load_embeddings(embeddings_filepath, source_vocab, dimension)
    print >> sys.stderr, "Loading source data..."
    S = trainer.load_data(source_filepath, source_vocab)
    print >> sys.stderr, "Loading target data..."
    T = trainer.load_data(target_filepath, target_vocab)

    print >> sys.stderr, "Optimizing..."
    # Initialize embeddings.
    K = P0.shape[1]
    N = S.shape[0]
    P = np.zeros_like(P0) # P0.copy()
    Q = np.zeros((len(target_vocab), K))
    DP_sqnorm = np.zeros_like(P)
    DQ_sqnorm = np.zeros_like(Q)
    trainer.compute_residuals(S, T, P, Q)
    for epoch in xrange(num_epochs):
        tic = time.time()
        trainer.compute_gradient(S, T, P0, mu, mu_s, mu_t, P, Q)
        DP_sqnorm += trainer.DP*trainer.DP
        DQ_sqnorm += trainer.DQ*trainer.DQ
        #pdb.set_trace()
        step = eta / (epsilon + np.sqrt(DP_sqnorm))
        P -= step * trainer.DP
        step = eta / (epsilon + np.sqrt(DQ_sqnorm))
        Q -= step * trainer.DQ

        trainer.compute_residuals(S, T, P, Q)
        if trainer.regularization_type == 'l2':
            parallel_reg = (0.5 * mu * np.linalg.norm(trainer.residuals.flatten())**2)/N
        else: # l1.
            parallel_reg = (mu * np.linalg.norm(trainer.residuals.flatten(), 1))/N
        loss_term = (0.5 * np.linalg.norm((P - P0).flatten())**2)/len(source_vocab)
        reg_s = 0.5 * mu_s * np.linalg.norm(P.flatten())**2
        reg_t = 0.5 * mu_t * np.linalg.norm(Q.flatten())**2
        loss = parallel_reg + loss_term + reg_s + reg_t
        gradient_norm = np.sqrt(np.linalg.norm(trainer.DP.flatten())**2 + \
                                np.linalg.norm(trainer.DQ.flatten())**2)
        toc = time.time()

        print >> sys.stderr, 'Epoch: %d, Loss: %f (%f+%f+%f+%f), Gradient: %f, Time: %f' % \
            (epoch, loss, parallel_reg, loss_term, reg_s, reg_t, gradient_norm, toc-tic)

    trainer.dump_embeddings(P, source_vocab, source_embeddings_filepath)
    trainer.dump_embeddings(Q, target_vocab, target_embeddings_filepath)
