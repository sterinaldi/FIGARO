import numpy as np
from numba import jit, njit, prange
from numba.extending import get_cython_function_address
import ctypes
from scipy.stats import invgamma, invwishart
from scipy.special import logsumexp

LOGSQRT2 = np.log(2*np.pi)

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
def log_add_array(x,y):
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
        res[i] = log_add(x[i],y[i])
    return res
    
@njit
def numba_gammaln(x):
    return gammaln_float64(x)

@jit
def log_invgamma(var, a, b):
    """
    Inverse Gamma logpdf
    
    Arguments:
        :double var: value
        :double a:   shape parameter
        :double b:   scale parameter
    
    Returns:
        :double: InverseGamma(a,b).logpdf(var^2)
    """
    return a*np.log(b) - (a+1)*np.log(var**2) - b/var**2 - numba_gammaln(a)

@jit
def log_norm_1d(x, m, s):
    """
    1D Normal logpdf
    
    Arguments:
        :double x: value
        :double m: mean
        :double s: std
    
    Returns:
        Normal(m,s).logpdf(x)
    """
    return -(x-m)**2/(2*s) - 0.5*np.log(2*np.pi) - 0.5*np.log(s)

@njit
def inv_jit(M):
  return np.linalg.inv(M)

@njit
def logdet_jit(M):
    return np.log(np.linalg.det(M))

@njit
def triple_product(v, M, n):
    """
    Triple product: v*M*v^T
    
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
    exponent = -0.5*triple_product(x-mu, inv_cov, len(mu))
    lognorm  = LOGSQRT2-0.5*logdet_jit(inv_cov)
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

def build_mean_cov(x, dim):
    """
    Build mean and covariance matrix from array.
    
    Arguments:
        :np.ndarray x: values for mean and covariance. Mean values are the first dim entries, stds are the second dim entries and off-diagonal elements are the remaining dim*(dim-1)/2 entries.
        :int dim:      number of dimensions
    
    Returns:
        :np.ndarray: mean
        :np.ndarray: covariance matrix
    """
    mean  = x[:dim]
    corr  = np.identity(dim)/2.
    corr[np.triu_indices(dim, 1)] = x[2*dim:]
    corr  = corr + corr.T
    sigma = x[dim:2*dim]
    cov_mat = np.multiply(corr, np.outer(sigma, sigma))
    return mean, cov_mat

#------------#
# 1D methods #
#------------#

@jit
def propose_point_1d(old_point, dm, ds):
    """
    Propose a new point uniformly drawn in an interval [x-dx, x+dx]
    
    Arguments:
        :np.ndarray old_point: old mean and std
        :double dm:            interval width for mean
        :double ds:            interval width for std
    
    Returns:
        :np.ndarray: new point
    """
    m = old_point[0] + (np.random.rand() - 0.5)*2*dm
    s = np.exp(np.log(old_point[1]) + (np.random.rand() - 0.5)*2*ds)
    return np.array([m,s])

#@jit
def sample_point_1d(means, covs, log_w, burnin = 1000, dm = 1, ds = 0.05, a = 2, b = 0.2):
    """
    1D metropolis sampling scheme to find maximum a posteriori for component mean and covariance
    
    Arguments:
        :iterable means: container for means of every event associated with the component (3d array)
        :iterable covs:  container for variances of every event associated with the component (4d array)
        :iterable log_w: container for weights of every event associated with the component (2d array)
        :int burnin:     number of iteration before returning the sampled point
        :double dm:      interval width for mean
        :double ds:      interval width for std
        :double a:       Inverse Gamma prior shape parameter (std)
        :double b:       Inverse Gamma prior scale parameter (std)
    
    Returns:
        :np.ndarray: array storing sampled mean and std
    """
    old_point = np.array([0., b])
    log_old = log_integrand_1d(old_point[0], old_point[1], means, covs, log_w, a, b)
    for i in range(burnin):
        new_point = propose_point_1d(old_point, dm, ds)
        log_new = log_integrand_1d(new_point[0], new_point[1], means, covs, log_w, a, b)
        if log_new > log_old:
            old_point = new_point
            log_old   = log_new
    return old_point

#@jit
def log_integrand_1d(mu, sigma, means, covs, log_w, a, b):
    """
    Probability distribution for mean and std - Eq. (46) with Inverse Gamma prior on std
    
    Arguments:
        :double mu:      temptative mean
        :double sigma:   temptative std
        :iterable means: container for means of every event associated with the component (3d array)
        :iterable covs:  container for variances of every event associated with the component (4d array)
        :iterable log_w: container for weights of every event associated with the component (2d array)
        :double a:       Inverse Gamma prior shape parameter (std)
        :double b:       Inverse Gamma prior scale parameter (std)
    
    Returns:
        :double: log probability
    """
    logP = 0.
    for i in range(len(means)):
        logP += log_prob_mixture_1d(mu, sigma, log_w[i], means[i], covs[i])
    return logP + log_invgamma(sigma, a, b)

@jit
def log_prob_mixture_1d(mu, sigma, log_w, means, covs):
    """
    Single term of productory in Eq. (46) - Single event.
    
    Arguments:
        :double mu:        temptative mean
        :double sigma:     temptative std
        :np.ndarray log_w: component weights
        :np.ndarray means: component means (2d array)
        :np.ndarray covs:  component variances (3d array)
    
    Returns:
        :double: log probability
    """
    logP = -np.inf
    for i in prange(len(means)):
        logP = log_add(logP, log_w[i] + log_norm_1d(means[i,0], mu, sigma**2 + covs[i,0,0] + (means[i,0] - mu)**2))
    return logP

def MC_predictive_1d(events, n_samps = 1000, m_min = -7, m_max = 7, a = 2, b = 0.2):
    """
    Monte Carlo integration over mean and std of p(m,s|{y}) - 1D
    
    Arguments:
        :iterable events: container of mixture instances (see mixture.py for the definition)
        :int n_samps:     number of MC draws
        :double m_min:    lower bound for uniform mean distribution
        :double m_max:    upper bound for uniform mean distribution
        :double a:        Inverse Gamma prior shape parameter (std)
        :double b:        Inverse Gamma prior scale parameter (std)
    
    Returns:
        :double: MC estimate of integral
    """
    means = np.random.uniform(m_min, m_max, size = n_samps)
    variances = np.sqrt(invgamma(a, b).rvs(size = n_samps))
    logP = np.zeros(n_samps, dtype = np.float64)
    for ev in events:
        logP += log_prob_mixture_1d_MC(means, variances, ev.log_w, ev.means, ev.covs)
    logP = logsumexp(logP)
    return logP - np.log(n_samps)

@jit
def log_prob_mixture_1d_MC(mu, sigma, log_w, means, covs):
    """
    Log probability for a single event - 1D
    
    Arguments:
        :np.ndarray mu:    array of temptative means
        :np.ndarray sigma: array of temptative stds
        :np.ndarray log_w: component weights
        :np.ndarray means: component means (2d array)
        :np.ndarray covs:  component variances (3d array)
    
    Returns:
        :np.ndarray: log probabilities for each pair of temptative mean and std
    """
    logP = -np.ones(len(mu), dtype = np.float64)*np.inf
    for i in prange(len(means)):
        logP = log_add_array(logP, log_w[i] + log_norm_1d(means[i,0], mu, sigma**2 + covs[i,0,0] + (means[i,0] - mu)**2))
    return logP

#------------#
# ND methods #
#------------#

@jit
def propose_point(old_point, dm, ds, dr, dim):
    """
    Propose a new point uniformly drawn in an interval [x-dx, x+dx]
    
    Arguments:
        :np.ndarray old_point: old mean and covariance
        :double dm:            interval width for mean
        :double ds:            interval width for covariance matrix diagonal elements
        :double dr:            interval width for covariance matrix off-diagonal elements
        :int dim:              number of dimensions
    
    Returns:
        :np.ndarray: new point
    """
    m = [old_point[i] + (np.random.rand() - 0.5)*2*dm for i in range(dim)]
    s = [old_point[i+dim] + (np.random.rand() - 0.5)*2*ds for i in range(dim)]
    r = [old_point[i+2*dim] + (np.random.rand() - 0.5)*2*dr for i in range(int(dim*(dim-1)/2.))]
    return np.array(m+r+s)

def sample_point(means, covs, log_w, dim, burnin = 1000, dm = 1, ds = 0.05, dr = 0.05, a = 2, b = 0.2**2):
    """
    Multidimensional metropolis sampling scheme to find maximum a posteriori for component mean and covariance matrix
    
    Arguments:
        :iterable means: container for means of every event associated with the component (3d array)
        :iterable covs:  container for variances of every event associated with the component (4d array)
        :iterable log_w: container for weights of every event associated with the component (2d array)
        :int dim:        number of dimensions
        :int burnin:     number of iteration before returning the sampled point
        :double dm:      interval width for mean
        :double ds:      interval width for diagonal elements
        :double dr:      interval width for off-diagonal elements
        :double a:       Inverse Wishart prior shape parameter (std)
        :double b:       Inverse Wishart prior scale matrix (std)
    
    Returns:
        :np.ndarray: array storing sampled mean and covariance matrix
    """
    mu = np.zeros(dim)
    if type(b) is float:
        cov = np.identity(dim)*b
    else:
        cov = b
    prior = invwishart(a, cov)
    old_point = np.concatenate((mu, [cov[i,i] for i in range(dim)], np.zeros(int(dim*(dim-1)/2.))))
    log_old = log_integrand(mu, cov, means, covs, log_w) + prior.logpdf(cov)
    for i in range(burnin):
        new_point = propose_point(old_point, dm, ds, dr, dim)
        if (new_point[2*dim:] > -1).all() and (new_point[2*dim:] < 1).all() and (new_point[dim:2*dim] > 0.).all():
            mu, cov = build_mean_cov(new_point, dim)
            log_new = log_integrand(mu, cov, means, covs, log_w) + prior.logpdf(cov)
            if log_new > log_old:
                old_point = new_point
                log_old   = log_new
    return old_point

#@jit
def log_integrand(mu, sigma, means, covs, log_w):
    """
    Probability distribution for mean and std - Eq. (46) with Inverse Wishart prior on std
    
    Arguments:
        :double mu:      temptative mean
        :double sigma:   temptative covariance matrix
        :iterable means: container for means of every event associated with the component (3d array)
        :iterable covs:  container for covariances of every event associated with the component (4d array)
        :iterable log_w: container for weights of every event associated with the component (2d array)
    
    Returns:
        :double: log probability
    """
    logP = 0.
    for i in range(len(means)):
        logP += log_prob_mixture(mu, sigma, means[i], covs[i], log_w[i])
    return logP

@jit
def log_prob_mixture(mu, cov, means, sigmas, log_w):
    """
    Single term of productory in Eq. (46) - Single event, multidimensional.
    
    Arguments:
        :double mu:        temptative mean
        :double sigma:     temptative covariance matrix
        :np.ndarray means: component means (2d array)
        :np.ndarray covs:  component covariances (3d array)
        :np.ndarray log_w: component weights
    
    Returns:
        :double: log probability
    """
    logP = -np.inf
    for i in range(len(means)):
        logP = log_add(logP, log_w[i] + log_norm(means[i], mu, sigmas[i] + cov + (means[i] - mu).T@(means[i] - mu)))
    return logP

def MC_predictive(events, dim, n_samps = 1000, m_min = -7, m_max = 7, a = 2, b = np.array([0.2])):
    """
    Monte Carlo integration over mean and std of p(m,s|{y}) - multidimensional
    
    Arguments:
        :iterable events: container of mixture instances (see mixture.py for the definition)
        :int dim:         number of dimensions
        :int n_samps:     number of MC draws
        :double m_min:    lower bound for uniform mean distribution
        :double m_max:    upper bound for uniform mean distribution
        :double a:        Inverse Wishart prior shape parameter
        :double b:        Inverse Wishart prior scale matrix
    
    Returns:
        :double: MC estimate of integral
    """
    means = np.random.uniform(m_min, m_max, size = (n_samps, dim))
    if len(b) == 1:
        b = np.identity(dim)*b
    variances = np.array(invwishart(a, b).rvs(size = n_samps))
    logP = np.zeros(n_samps, dtype = np.float64)
    for ev in events:
        logP += log_prob_mixture_MC(means, variances, ev.log_w, ev.means, ev.covs)
    logP = logsumexp(logP)
    return logP - np.log(n_samps)

@jit
def log_prob_mixture_MC(mu, cov, log_w, means, covs):
    """
    Log probability for a single event - multidimensional
    
    Arguments:
        :np.ndarray mu:    array of temptative means
        :np.ndarray sigma: array of temptative covariances
        :np.ndarray log_w: component weights
        :np.ndarray means: component means (2d array)
        :np.ndarray covs:  component covariances (3d array)
    
    Returns:
        :np.ndarray: log probabilities for each pair of temptative mean and std
    """
    logP = -np.ones(len(mu), dtype = np.float64)*np.inf
    for i in prange(len(means)):
        logP = log_add_array(logP, log_w[i] + log_norm_array(means[i], mu, covs[i] + cov + (means[i] - mu).T@(means[i] - mu)))
    return logP
