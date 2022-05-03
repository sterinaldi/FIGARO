import numpy as np
from numba import jit, njit, prange
from numba.extending import get_cython_function_address
import ctypes
from scipy.stats import invgamma, invwishart, multivariate_normal
from scipy.special import logsumexp

LOG2PI = np.log(2*np.pi)

#-----------#
# Functions #
#-----------#

@jit
def _log_add(x, y):
    """
    Compute log(np.exp(x) + np.exp(y))
    
    Arguments:
        :double x: first addend (log)
        :double y: second addend (log)
    
    Returns:
        :double: log(np.exp(x) + np.exp(y))
    """
    if x >= y:
        return x+np.log1p(np.exp(y-x))
    else:
        return y+np.log1p(np.exp(x-y))

@jit
def _log_add_array(x,y):
    """
    Compute log(np.exp(x) + np.exp(y)) element-wise
    
    Arguments:
        :np.ndarray x: first addend (log)
        :np.ndarray y: second addend (log)
    
    Returns:
        :np.ndarray: log(np.exp(x) + np.exp(y)) element-wise
    """
    res = np.zeros(len(x), dtype = np.float64)
    for i in prange(len(x)):
        res[i] = _log_add(x[i],y[i])
    return res

@jit
def log_norm_1d(x, m, s):
    """
    1D Normal logpdf
    
    Arguments:
        :double x: value
        :double m: mean
        :double s: var
    
    Returns:
        Normal(m,s).logpdf(x)
    """
    return -(x-m)**2/(2*s) - 0.5*np.log(2*np.pi*s)

@njit
def inv_jit(M):
  return np.linalg.inv(M)

@njit
def logdet_jit(M):
    return np.log(np.linalg.det(M))

@njit
def scalar_product(v, M, n):
    """
    Scalar product: v*M*v^T
    
    Arguments:
        :np.ndarray v: array
        :np.ndarray M: matrix
        :int n:        len(v)
    
    Returns:
        :double: v*M*v^T
    """
    res = 0.
    for i in prange(n):
        for j in prange(n):
            res = res + M[i,j]*v[i]*v[j]
    return res

@jit
def log_norm(x, mu, cov):
    """
    Multivariate Normal logpdf
    
    Arguments:
        :np.ndarray x: value
        :np.ndarray m: mean vector
        :np.ndarray s: covariance matrix
    
    Returns:
        :double: MultivariateNormal(m,s).logpdf(x)
    """
    inv_cov  = inv_jit(cov)
    exponent = -0.5*scalar_product(x-mu, inv_cov, len(mu))
    lognorm  = 0.5*len(mu)*LOG2PI+0.5*logdet_jit(cov)
    return -lognorm+exponent

@jit
def log_norm_array(x, mu, cov):
    """
    Multivariate Normal logpdf element-wise wrt mu and cov
    
    Arguments:
        :np.ndarray x: value
        :np.ndarray m: vector of mean vectors
        :np.ndarray s: vector of covariance matrices
    
    Returns:
        :np.ndarray: MultivariateNormal(m,s).logpdf(x)
    """
    vect = np.zeros(len(mu), dtype = np.float64)
    for i in prange(len(mu)):
        vect[i] = log_norm(x, mu[i], cov[i])
    return vect

#------------#
# 1D methods #
#------------#


def compute_probability_1d(mu, sigma, means, vars, log_w):
    logP = np.zeros(len(mu), dtype = np.float64)
    for i, (m, s) in enumerate(zip(mu, sigma)):
        logP[i] = eval_mix_1d(m, s, means, vars, log_w)
    return logP

def eval_mix_1d(mu, sigma, means, vars, log_w):
    return logsumexp([log_norm_1d(means[i,0], mu, sigma+vars[i,0,0]) for i in range(len(means))], b = np.exp(log_w))

#------------#
# ND methods #
#------------#

def compute_probability(mu, sigma, means, vars, log_w):
    logP = np.zeros(len(mu), dtype = np.float64)
    for i, (m, s) in enumerate(zip(mu, sigma)):
        logP[i] = eval_mix(m, s, means, vars, log_w)
    return logP

def eval_mix(mu, sigma, means, vars, log_w):
    return logsumexp([log_norm(means[i], mu, sigma+vars[i]) for i in range(len(means))], b = log_w)
