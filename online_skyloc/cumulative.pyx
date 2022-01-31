# encoding: utf-8
# cython: profile=False
# cython: linetrace=False
# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False, binding=True, embedsignature=True
from __future__ import division
import numpy as np
cimport numpy as np
from libc.math cimport log,exp

'''
From https://github.com/wdpozzo/3d_volume/cumulative.pyx
'''
#
#cdef extern from "math.h":
#    double log(double x)
#cdef extern from "math.h":
#    double exp(double x)

cdef inline double d_max(float a, float b) nogil: return a if a >= b else b
cdef inline double log_add(float x, float y) nogil: return x+log(1+exp(y-x)) if x >= y else y+log(1+exp(x-y))

def fast_log_cumulative(np.ndarray[float, ndim=1]  f):
    return _fast_log_cumulative(f)

cdef np.ndarray[float, ndim=1] _fast_log_cumulative(np.ndarray[float, ndim=1]  f):

    cdef unsigned int n = f.shape[0]
    cdef np.ndarray[float, ndim=1] h = np.zeros(n, dtype = np.float64)

    h[0] = f[0]
    for i in range(1,n):
        h[i] = log_add(h[i-1],f[i])
    
    cdef double hmax = h[n-1]
    
    for i in range(n):
        h[i] = h[i] - hmax
        
    return h
