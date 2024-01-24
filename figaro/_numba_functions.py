import numpy as np
from numba import njit, prange
from numba.extending import get_cython_function_address
import ctypes

@njit
def inv_jit(M):
  return np.linalg.inv(M)

@njit
def logdet_jit(M):
    return np.log(np.linalg.det(M))

@njit
def logsumexp_jit(a):
    a_max = np.max(a)
    return np.log(np.sum(np.exp(a - a_max))) + a_max

@njit
def logsumexp_jit_weighted(a, b):
    a_max = np.max(a)
    tmp = b * np.exp(a - a_max)
    return np.log(np.sum(tmp)) + a_max

@njit
def log_add(x, y):
    if x >= y:
        return x+log1p_jit(np.exp(y-x))
    else:
        return y+log1p_jit(np.exp(x-y))

@njit
def outer_jit(x, y):
    return np.outer(x, y)
    
@njit
def divide_jit(x, y):
    return np.divide(x, y)

@njit
def diag_jit(m):
    return np.diag(m)

@njit
def rescale_matrix(S, n):
    std = np.sqrt(diag_jit(S))
    rho = divide_jit(S, outer_jit(std,std))
    return rho * outer_jit(std/np.sqrt(n), std/np.sqrt(n))

@njit
def eigh_jit(m):
    return np.linalg.eigh(m)

@njit
def log1p_jit(x):
    return np.log1p(x)

"""
See https://stackoverflow.com/a/54855769
Wrapper (based on https://github.com/numba/numba/issues/3086) for scipy's cython implementation of gammaln.
"""

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
functype = ctypes.CFUNCTYPE(_dble, _dble)
gammaln_float64 = functype(addr)

@njit
def gammaln_jit(x):
    return gammaln_float64(x)
