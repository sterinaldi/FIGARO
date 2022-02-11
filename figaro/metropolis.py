import numpy as np
from numpy.random import uniform
from numba import jit, njit, prange
from numba.extending import get_cython_function_address
import ctypes

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
functype = ctypes.CFUNCTYPE(_dble, _dble)
gammaln_float64 = functype(addr)

@jit
def log_add(x, y):
     if x >= y:
        return x+np.log1p(np.exp(y-x))
     else:
        return y+np.log1p(np.exp(x-y))

@njit
def numba_gammaln(x):
    return gammaln_float64(x)

@jit
def propose_point(old_point, dm, ds):
    m = old_point[0] + (np.random.rand() - 0.5)*2*dm
    s = old_point[1] + (np.random.rand() - 0.5)*2*ds
    return np.array([m,s])

def sample_point(samples, m_min = -20, m_max = 20, s_min = 0, s_max = 1, burnin = 500, dm = 1, ds = 0.05, a = 1, b = 0.2):
    old_point = np.array([m_min + np.random.rand()*m_max, s_min + np.random.rand()*s_max])
    for i in range(burnin):
        new_point = propose_point(old_point, dm, ds)
        if not (s_min < new_point[1] < s_max and m_min < new_point[0] < m_max):
            log_new = -np.inf
            log_old = 0.
        else:
            log_new = log_integrand_1d(new_point[0], new_point[1], samples, a, b)
            log_old = log_integrand_1d(old_point[0], old_point[1], samples, a, b)
        if log_new > log_old:# > np.log(np.random.rand()):
            old_point = new_point
    return np.array(old_point)

@jit
def log_integrand_1d(mu, sigma, samples, a, b):
    logP = -np.inf
    for i in prange(len(samples)):
        ss = samples[i].flatten()
        logP = log_add(logP, log_norm(ss, mu, sigma).sum())
    return logP  - np.log(len(samples)) + log_invgamma(sigma, a, b)

@jit
def log_invgamma(var, a, b):
    return a*np.log(b) - (a+1)*np.log(var**2) - b/var**2 - numba_gammaln(a)

@jit
def log_norm(x, m, s):
    return -(x-m)**2/(2*s*s) - 0.5*np.log(2*np.pi) - np.log(s)
