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
def logsumexp_jit(a, b):
    a_max = np.max(a)
    tmp = b * np.exp(a - a_max)
    return np.log(np.sum(tmp)) + a_max

@njit
def log_add(x, y):
    if x >= y:
        return x+_log1p_jit(np.exp(y-x))
    else:
        return y+_log1p_jit(np.exp(x-y))

@njit
def _outer_jit(x, y):
    return np.outer(x, y)
    
@njit
def _divide_jit(x, y):
    return np.divide(x, y)

@njit
def _diag_jit(m):
    return np.diag(m)

@njit
def _rescale_matrix(S, n):
    std = np.sqrt(_diag_jit(S))
    rho = _divide_jit(S, _outer_jit(std,std))
    return rho * _outer_jit(std/np.sqrt(n), std/np.sqrt(n))

@njit
def _eigh_jit(m):
    return np.linalg.eigh(m)

@njit
def _log1p_jit(x):
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
def _gammaln_jit(x):
    return gammaln_float64(x)
