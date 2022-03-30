import numpy as np
import cpnest.model
from numba import jit, njit, prange
from numba.extending import get_cython_function_address
from numba.typed import List
import ctypes
from scipy.stats import invgamma, invwishart
from scipy.special import logsumexp

#-----------#
# Utilities #
#-----------#

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
functype = ctypes.CFUNCTYPE(_dble, _dble)
gammaln_float64 = functype(addr)

#-----------#
# Functions #
#-----------#

@jit
def log_add(x, y):
     if x >= y:
        return x+np.log1p(np.exp(y-x))
     else:
        return y+np.log1p(np.exp(x-y))

@jit
def log_add_array(x,y):
    res = np.zeros(len(x), dtype = np.float64)
    for i in prange(len(x)):
        res[i] = log_add(x[i],y[i])
    return res
    
@njit
def numba_gammaln(x):
    return gammaln_float64(x)

@jit
def log_invgamma(var, a, b):
    return a*np.log(b) - (a+1)*np.log(var**2) - b/var**2 - numba_gammaln(a)

@njit
def inv_jit(M):
  return np.linalg.inv(M)

@jit
def inv_vect_jit(Ms):
    vect = np.zeros(Ms.shape, dtype = np.float64)
    for i in prange(Ms.shape[0]):
        vect[i] = inv_jit(Ms[i])
    return vect

@njit
def logdet_jit(M):
    return np.log(np.linalg.det(M))

@jit
def logdet_vect_jit(Ms):
    vect = np.zeros(Ms.shape[0], dtype = np.float64)
    for i in prange(Ms.shape[0]):
        vect[i] = logdet_jit(Ms[i])
    return vect

@njit
def triple_product(v, M, n):
    res = 0.
    for i in prange(n):
        for j in prange(n):
            res = res + M[i,j]*v[i]*v[j]
    return res

@jit
def triple_product_vect(v, M, n):
    vect = np.zeros(v.shape[0], dtype = np.float64)
    for i in prange(v.shape[0]):
        vect[i] = triple_product(v[i], M[i], n)
    return vect

@jit
def log_norm(x, mu, cov):
    inv_cov  = inv_vect_jit(cov)
    exponent = -0.5*triple_product_vect(x - mu, inv_cov, x.shape[0])
    lognorm  = 0.5*np.log(2*np.pi)-0.5*logdet_vect_jit(inv_cov)
    return -lognorm+exponent

@jit
def log_norm_1d(x, m, s):
    return -(x-m)**2/(2*s) - 0.5*np.log(2*np.pi) - 0.5*np.log(s)


@jit
def build_mean_cov(x, dim):
    mean  = x[:dim]
    corr  = np.identity(dim)/2.
    ctr = 0
    for i in prange(dim-1):
        for j in prange(i+1, dim):
            corr[i][j] = x[2*dim:][ctr]
            ctr = ctr+1
    corr  = corr + corr.T
    sigma = x[dim:2*dim]
    cov_mat = np.multiply(corr, np.outer(sigma, sigma))
    return mean, cov_mat

#---------------#
# 1D Metropolis #
#---------------#

@jit
def propose_point_1d(old_point, dm, ds):
    m = old_point[0] + (np.random.rand() - 0.5)*2*dm
    s = old_point[1] + (np.random.rand() - 0.5)*2*ds
    return np.array([m,s])

#@jit
def sample_point_1d(means, covs, log_w, m_min = -20, m_max = 20, s_min = 0, s_max = 1, burnin = 1000, dm = 1, ds = 0.05, a = 2, b = 0.2):
    old_point = np.array([0, 0.02])
    log_old = log_integrand_1d(old_point[0], old_point[1], means, covs, log_w, a, b)
    for i in range(burnin):
        new_point = propose_point(old_point, dm, ds)
        if not (s_min < new_point[1] < s_max and m_min < new_point[0] < m_max):
            log_new = -np.inf
        else:
            log_new = log_integrand_1d(new_point[0], new_point[1], means, covs, log_w, a, b)
        if log_new > log_old:
            old_point = new_point
            log_old   = log_new
    return old_point

@jit
def log_integrand_1d(mu, sigma, means, covs, log_w, a, b):
    logP = 0
    for i in range(len(means)):
        logP += log_prob_mixture_1d(mu, sigma, log_w[i], means[i], covs[i])
    return logP + log_invgamma(sigma, a, b)

@jit
def log_prob_mixture_1d(mu, sigma, log_w, means, covs):
    logP = -np.inf
    for i in prange(len(means)):
        logP = log_add(logP, log_w[i] + log_norm_1d(means[i][0], mu, sigma**2 + covs[i][0,0] + (means[i][0] - mu)**2))
    return logP

def MC_predictive_1d(events, n_samps = 1000, m_min = -5, m_max = 5, a = 2, b = 0.2):
    means = np.random.uniform(m_min, m_max, size = n_samps)
    variances = np.sqrt(invgamma(a, b).rvs(size = n_samps))
    logP = np.zeros(n_samps, dtype = np.float64)
    for ev in events:
        logP += log_prob_mixture_1d_MC(means, variances, ev.log_w, ev.means, ev.covs)
    logP = logsumexp(logP)
    return logP - np.log(n_samps)

@jit
def log_prob_mixture_1d_MC(mu, sigma, log_w, means, covs):
    logP = -np.ones(len(mu), dtype = np.float64)*np.inf
    for i in prange(len(means)):
        logP = log_add_array(logP, log_w[i] + log_norm_1d(means[i][0], mu, sigma**2 + covs[i][0,0] + (means[i][0] - mu)**2))
    return logP

#---------------#
# ND Metropolis #
#---------------#

#@jit
def log_integrand(x, means, sigmas, log_w, dim, df, L):
    mu, cov = build_mean_cov(x, dim)
    log_prior = invwishart(df = df, scale = L).logpdf(cov)
    mu  = np.atleast_2d([mu])
    cov = np.atleast_3d([cov])
    logP = 0.
    for i in prange(len(means)):
        logP_i = -np.inf
        for j in prange(len(means[i])):
            logP_i = log_add(logP_i, log_w[i][j] + log_norm(means[i][j], mu, sigmas[i][j] + cov + (means[i][j] - mu).T@(means[i][j] - mu))[0])
        logP = logP + logP_i
    return logP + log_prior

@jit
def propose_point(old_point, dm, ds, dim):
    m = [old_point[i] + (np.random.rand() - 0.5)*2*dm for i in range(dim)]
    s = [old_point[dim + i] + (np.random.rand() - 0.5)*2*ds for i in range(dim)]
    r = [old_point[2*dim + i] + (np.random.rand() - 0.5)*2*ds for i in range(int(dim*(dim-1)/2.))]
    return np.array(m+s+r)

#@jit
def sample_point(means, covs, log_w, dim, df, L, m_min = -20, m_max = 20, s_min = 0, s_max = 1, burnin = 1000, dm = 1, ds = 0.05):
    m = [0. for i in range(dim)]
    s = [L[i,i] for i in range(dim)]
    r = [0. for i in range(int(dim*(dim-1)/2.))]
    old_point = np.array(m+s+r)
    log_old = log_integrand(old_point, means, covs, log_w, dim, df, L)
    for i in range(burnin):
        new_point = propose_point(old_point, dm, ds, dim)
        if not ((s_min < new_point[dim:2*dim]).all() and (new_point[dim:2*dim] < s_max).all() and (m_min < new_point[:dim]).all() and (new_point[:dim] < m_max).all() and (-1 < new_point[:2*dim]).all() and (new_point[:2*dim] < 1).all()):
            log_new = -np.inf
        else:
            log_new = log_integrand(new_point, means, covs, log_w, dim, df, L)
        if log_new > log_old:
            old_point = new_point
            log_old   = log_new
    return old_point

def MC_predictive(events, dim, n_samps = 1000, m_min = -5, m_max = 5, a = 2, b = np.array([0.2])):
    means = np.random.uniform(m_min, m_max, size = (n_samps, dim))
    if len(b) == 1:
        b = np.identity(dim)*b
    variances = invwishart(a, b).rvs(size = n_samps)
    logP = np.zeros(n_samps, dtype = np.float64)
    for ev in events:
        logP += log_prob_mixture_MC(means, variances, ev.log_w, ev.means, ev.covs)
    logP = logsumexp(logP)
    return logP - np.log(n_samps)

@jit
def log_prob_mixture_MC(mu, sigma, log_w, means, covs):
    logP = -np.ones(len(mu), dtype = np.float64)*np.inf
    for i in prange(len(means)):
        logP = log_add_array(logP, log_w[i] + log_norm(means[i], mu, covs[i] + sigma + (means[i] - mu).T@(means[i] - mu)))
    return logP
