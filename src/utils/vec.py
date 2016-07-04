import numpy as np
from decimal import *

def exp_vec(x):
    return np.exp(Decimal(x))
exp_vec = np.vectorize(exp_vec, otypes=[np.dtype(Decimal)])
def ln_vec(x):
    return float(x.ln())
ln_vec = np.vectorize(ln_vec, otypes=[np.float])
