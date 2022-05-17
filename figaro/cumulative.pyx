# encoding: utf-8
# cython: profile=False
# cython: linetrace=False
# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False, binding=True, embedsignature=True
import numpy as np
cimport numpy as np
from libc.math cimport log,exp

'''
From https://github.com/wdpozzo/3d_volume/cumulative.pyx
'''

cdef inline double log_add(double x, double y) nogil: return x+log(1+exp(y-x)) if x >= y else y+log(1+exp(x-y))

def fast_log_cumulative(np.ndarray[double, ndim=1, mode="c"]  f):
    """
    Compute log cdf of probability density f
    
    Arguments:
        :np.ndarray f: probability density
    
    Returns:
        :np.ndarray: log cdf
    """
    return _fast_log_cumulative(f)

cdef np.ndarray[double, ndim=1, mode="c"] _fast_log_cumulative(np.ndarray[double, ndim=1, mode="c"] f):
    """
    Compute log cdf of probability density f
    
    Arguments:
        :np.ndarray f: probability density
    
    Returns:
        :np.ndarray: log cdf
    """
    cdef unsigned int n = f.shape[0]
    cdef np.ndarray[double, ndim=1, mode="c"] h = np.zeros(n, dtype = np.double)
    h[0] = f[0]
    for i in range(1,n):
        h[i] = log_add(h[i-1],f[i])
    return h

def fast_cumulative(np.ndarray[double, ndim=1, mode="c"]  f):
    """
    Compute cdf of probability density f
    
    Arguments:
        :np.ndarray f: probability density
    
    Returns:
        :np.ndarray: cdf
    """
    return _fast_cumulative(f)

cdef np.ndarray[double, ndim=1, mode="c"] _fast_cumulative(np.ndarray[double, ndim=1, mode="c"] f):
    """
    Compute cdf of probability density f
    
    Arguments:
        :np.ndarray f: probability density
    
    Returns:
        :np.ndarray: cdf
    """
    cdef unsigned int n = f.shape[0]
    cdef np.ndarray[double, ndim=1, mode="c"] h = np.zeros(n, dtype = np.double)
    h[0] = f[0]
    for i in range(1,n):
        h[i] = h[i-1] + f[i]
    return h
