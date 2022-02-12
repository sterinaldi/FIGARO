import numpy as np
import cpnest.model
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

@jit
def sample_point(means, sigmas, m_min = -20, m_max = 20, s_min = 0, s_max = 1, burnin = 1000, dm = 1, ds = 0.05, a = 1, b = 0.2):
    old_point = np.array([m_min + np.random.rand()*m_max, s_min + np.random.rand()*s_max])
    for i in range(burnin):
        new_point = propose_point(old_point, dm, ds)
        if not (s_min < new_point[1] < s_max and m_min < new_point[0] < m_max):
            log_new = -np.inf
            log_old = 0.
        else:
            log_new = log_integrand_1d(new_point[0], new_point[1], means, sigmas, a, b)
            log_old = log_integrand_1d(old_point[0], old_point[1], means, sigmas, a, b)
        if log_new > log_old:
            old_point = new_point
    return old_point

@jit
def log_integrand_1d(mu, sigma, means, sigmas, a, b):
    logP = -np.inf
    for i in prange(len(means)):
        logP = log_add(logP, log_norm_1d(means[i][0], mu, np.sqrt(sigmas[i] + sigma**2)))
    return logP + log_invgamma(sigma, a, b)

@jit
def log_invgamma(var, a, b):
    return a*np.log(b) - (a+1)*np.log(var**2) - b/var**2 - numba_gammaln(a)

@jit
def log_norm_1d(x, m, s):
    return -(x-m)**2/(2*s*s) - 0.5*np.log(2*np.pi) - np.log(s)

class Integrator(cpnest.model.Model):
    
    def __init__(self, means, covs, dim, df, L):
        super(Integrator, self).__init__()
        self.means     = means
        self.covs      = covs
        self.dim       = dim
        self.names     = ['m{0}'.format(i+1) for i in range(self.dim)] + ['s{0}'.format(i+1) for i in range(self.dim)] + ['r{0}'.format(j) for j in range(int(self.dim*(self.dim-1)/2.))]
        self.bounds    = [[-20, 20] for _ in range(self.dim)] + [[0, 1] for _ in range(self.dim)] + [[-1,1] for _ in range(int(self.dim*(self.dim-1)/2.))]
        self.prior     = invwishart(df = df, scale = L)
        self.mu_prior  = -self.dim*np.log(40)
    
    def log_prior(self, x):
        logP = super(Integrator, self).log_prior(x)
        if not np.isfinite(logP):
            return -np.inf
        self.mean, self.cov_mat = build_mean_cov(np.array(x.values), self.dim)
        if not np.isfinite(logdet_jit(self.cov_mat)):
            return -np.inf
        logP = self.prior.logpdf(self.cov_mat) + self.mu_prior
        return logP
    
    def log_likelihood(self, x):
        return log_integrand(self.mean[0], self.cov_mat, self.means, self.covs)

def build_mean_cov(x, dim):
    mean  = np.atleast_2d(x[:dim])
    corr  = np.identity(dim)/2.
    corr[np.triu_indices(dim, 1)] = x[2*dim:]
    corr  = corr + corr.T
    sigma = np.identity(dim)*x[dim:2*dim]**2
    cov_mat = sigma@corr
    return mean, cov_mat

@njit
def inv_jit(M):
  return np.linalg.inv(M)

@njit
def logdet_jit(M):
    return np.log(np.linalg.det(M))

@njit
def triple_product(v, M, n):
    res = np.zeros(1, dtype = np.float64)
    for i in prange(n):
        for j in prange(n):
            res = res + M[i,j]*v[i]*v[j]
    return res

@jit
def log_norm(x, mu, cov):
    inv_cov  = inv_jit(cov)
    exponent = -0.5*triple_product(x, mu, inv_cov)
    lognorm  = LOGSQRT2-0.5*logdet_jit(inv_cov)
    return -lognorm+exponent

@jit
def log_integrand(mu, cov, means, sigmas):
    logP = -np.inf
    for i in prange(len(means)):
        logP = log_add(logP, log_norm(means[i], mu, sigmas[i] + cov))
    return logP
