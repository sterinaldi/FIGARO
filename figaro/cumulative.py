import numpy as np
from numba import njit, prange
from figaro._numba_functions import log_add

@njit
def fast_log_cumulative(f):
    """
    Compute log cdf of probability density f
    
    Arguments:
        :np.ndarray f: probability density
    
    Returns:
        :np.ndarray: log cdf
    """
    n = f.shape[0]
    h = np.zeros(n, dtype = np.double)
    h[0] = f[0]
    for i in prange(1,n):
        h[i] = log_add(h[i-1],f[i])
    return h

@njit
def fast_cumulative(f):
    """
    Compute cdf of probability density f
    
    Arguments:
        :np.ndarray f: probability density
    
    Returns:
        :np.ndarray: cdf
    """
    n = f.shape[0]
    h = np.zeros(n, dtype = np.double)
    h[0] = f[0]
    for i in prange(1,n):
        h[i] = h[i-1] + f[i]
    return h
