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

def expected_vals_MC_1d(means, covs, log_w, n_samps = 1000, a = 2, b = 0.2):
    """
    Computes the expected values for mean and variance via Monte Carlo integration
    
    Arguments:
        :iterable means: container for means of every event associated with the component (3d array)
        :iterable covs:  container for variances of every event associated with the component (4d array)
        :iterable log_w: container for weights of every event associated with the component (2d array)
        :int n_samps:    number of MC draws
        :double a:       Inverse Gamma prior shape parameter (std)
        :double b:       Inverse Gamma prior scale parameter (std)
    
    Returns:
        :np.ndarray: mean
        :np.ndarray: variance
    """
    mu = np.random.normal(size = n_samps)
    sigma = invgamma(a, scale = b).rvs(size = n_samps)
    P = np.zeros(n_samps, dtype = np.float64)
    for i in range(n_samps):
        P[i] = log_integrand_1d(mu[i], sigma[i], means, covs, log_w)
    P = np.exp(P)
    norm = np.sum(P)
    return np.atleast_2d(np.average(mu, weights = P/norm, axis = 0)), np.atleast_2d(np.average(sigma, weights = P/norm, axis = 0))

def log_integrand_1d(mu, sigma, means, covs, log_w):
    """
    Probability distribution for mean and variance
    
    Arguments:
        :double mu:      temptative mean
        :double sigma:   temptative std
        :iterable means: container for means of every event associated with the component (3d array)
        :iterable covs:  container for variances of every event associated with the component (4d array)
        :iterable log_w: container for weights of every event associated with the component (2d array)
    
    Returns:
        :double: log probability
    """
    logP = 0.
    for i in range(len(means)):
        logP += log_prob_mixture_1d(mu, sigma, log_w[i], means[i], covs[i])
    return logP

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
        logP = _log_add(logP, log_w[i] + log_norm_1d(means[i,0], mu, sigma + covs[i,0,0]))
    return logP

def MC_predictive_1d(events, n_samps = 1000, a = 2, b = 0.2):
    """
    Monte Carlo integration over mean and std of p(m,s|{y}) - 1D
    
    Arguments:
        :iterable events: container of mixture instances (see mixture.py for the definition)
        :int n_samps:     number of MC draws
        :double a:        Inverse Gamma prior shape parameter
        :double b:        Inverse Gamma prior scale parameter
    
    Returns:
        :double: MC estimate of integral
    """
    means = np.random.normal(size = n_samps)
    variances = invgamma(a, scale = b).rvs(size = n_samps)
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
        logP = _log_add_array(logP, log_w[i] + log_norm_1d(means[i,0], mu, sigma + covs[i,0,0]))
    return logP

#------------#
# ND methods #
#------------#

def expected_vals_MC(means, covs, log_w, dim, n_samps = 1000, a = 2, b = np.array([0.2])):
    """
    Computes the expected values for mean vector and covariance matrix via Monte Carlo integration
    
    Arguments:
        :iterable means: container for means of every event associated with the component (3d array)
        :iterable covs:  container for variances of every event associated with the component (4d array)
        :iterable log_w: container for weights of every event associated with the component (2d array)
        :int n_samps:    number of MC draws
        :double a:       Inverse Wishart prior shape parameter
        :double b:       Inverse Wishart prior scale matrix
    
    Returns:
        :np.ndarray: mean vector
        :np.ndarray: covariance matrix
    """
    mu = multivariate_normal(np.zeros(dim), np.identity(dim)).rvs(size = n_samps)
    if len(b) == 1:
        b = np.identity(dim)*b[0]
    sigma  = invwishart(df = np.max([a,dim]), scale = b).rvs(size = n_samps)
    P = np.zeros(n_samps, dtype = np.float64)
    for i in range(n_samps):
        P[i] = log_integrand(mu[i], sigma[i], means, covs, log_w)
    P = np.exp(P)
    return np.atleast_2d(np.average(mu, weights = P, axis = 0)), np.atleast_2d(np.average(sigma, weights = P, axis = 0))

def log_integrand(mu, sigma, means, covs, log_w):
    """
    Probability distribution for mean and covariance
    
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
        logP = _log_add(logP, log_w[i] + log_norm(means[i], mu, sigmas[i] + cov))
    return logP

def MC_predictive(events, dim, n_samps = 1000, a = 2, b = np.array([0.2])):
    """
    Monte Carlo integration over mean and std of p(m,s|{y}) - multidimensional
    
    Arguments:
        :iterable events: container of mixture instances (see mixture.py for the definition)
        :int dim:         number of dimensions
        :int n_samps:     number of MC draws
        :double a:        Inverse Wishart prior shape parameter
        :double b:        Inverse Wishart prior scale matrix
    
    Returns:
        :double: MC estimate of integral
    """
    means = multivariate_normal(np.zeros(dim), np.identity(dim)).rvs(size = n_samps)
    if len(b) == 1:
        b = np.identity(dim)*b[0]
    variances = invwishart(df = np.max([a,dim]), scale = b).rvs(size = n_samps)
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
        logP = _log_add_array(logP, log_w[i] + log_norm_array(means[i], mu, covs[i] + cov))
    return logP
