from __future__ import division
cimport numpy as np
import numpy as np
from libc.math cimport log, sqrt, M_PI, exp, HUGE_VAL, atan2, acos, sin, cos
cimport cython
from numpy.linalg import det, inv
from scipy.stats import multivariate_normal as mn

cdef double LOGSQRT2 = log(sqrt(2*M_PI))

cdef inline double log_add(double x, double y) nogil: return x+log(1.0+exp(y-x)) if x >= y else y+log(1.0+exp(x-y))
cdef inline double _scalar_log_norm(double x, double x0, double s) nogil: return -(x-x0)*(x-x0)/(2*s*s) - LOGSQRT2 - 0.5*log(s)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _triple_product(np.ndarray[double, ndim = 1, mode = "c"] x, np.ndarray[double, ndim = 1, mode = "c"] mu, np.ndarray[double, ndim = 2, mode = "c"] inv_cov):
    cdef unsigned int i,j
    cdef unsigned int n = x.shape[0]
    cdef double res     = 0.0
    for i in range(n):
        for j in range(n):
            res += inv_cov[i,j]*(x[i]-mu[i])*(x[j]-mu[j])
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _log_norm(np.ndarray[double, ndim = 1, mode = "c"] x, np.ndarray[double, ndim = 1, mode = "c"] mu, np.ndarray[double, ndim = 2, mode = "c"] cov):
    cdef np.ndarray inv_cov  = np.linalg.inv(cov)
    cdef double exponent     = -0.5*_triple_product(x, mu, inv_cov)
    cdef double lognorm      = LOGSQRT2-0.5*np.linalg.slogdet(inv_cov)[1]
    return -lognorm+exponent
             
def log_norm(np.ndarray[double, ndim = 1, mode = "c"] x, np.ndarray[double, ndim = 1, mode = "c"] x0, np.ndarray[double, ndim = 2, mode = "c"] sigma):
    return _log_norm(x, x0, sigma)

def scalar_log_norm(double x, double x0, double s):
    return _scalar_log_norm(x,x0,s)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _log_prob_component(np.ndarray[double, ndim = 1, mode = "c"] mu, np.ndarray[double, ndim = 1, mode = "c"] mean, np.ndarray[double, ndim = 2, mode = "c"] sigma, double w):
    return log(w) + _log_norm(mu, mean, sigma)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _log_prob_mixture(np.ndarray[double, ndim = 1, mode = "c"] mu, np.ndarray[double, ndim = 2, mode = "c"] sigma, object ev):
    cdef double logP = -HUGE_VAL
    for comp_mean, comp_weight in zip(ev.means, ev.w):
        logP = log_add(logP, _log_prob_component(mu, comp_mean, sigma, comp_weight))
    return logP

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _integrand(np.ndarray[double, ndim = 1, mode = "c"] mean, np.ndarray[double, ndim = 2, mode = "c"] covariance, list events, unsigned int dim):
    cdef unsigned int i,j
    cdef double logprob = 0.0
    cdef object ev
    for ev in events:
        logprob += _log_prob_mixture(mean, covariance, ev)
    return logprob

def integrand(np.ndarray[double, ndim = 1, mode = "c"] mean, np.ndarray[double, ndim = 2, mode = "c"] covariance, list events, unsigned int dim):
    return _integrand(mean, covariance, events, dim)

