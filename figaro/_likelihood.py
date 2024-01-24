import numpy as np
from figaro._numba_functions import *

LOG2PI = np.log(2*np.pi)

@njit
def scalar_product(v, M, n):
    """
    Scalar product: v*M*v^T
    
    Arguments:
        np.ndarray v: array
        np.ndarray M: matrix
        int n:        len(v)
    
    Returns:
        double: v*M*v^T
    """
    res = 0.
    for i in prange(n):
        for j in prange(n):
            res = res + M[i,j]*v[i]*v[j]
    return res

@njit
def log_norm_1d(x, m, s):
    """
    1D Normal logpdf
    
    Arguments:
        double x: value
        double m: mean
        double s: var
    
    Returns:
        Normal(m,s).logpdf(x)
    """
    return -(x-m)**2/(2*s) - 0.5*np.log(2*np.pi*s)

@njit
def log_norm(x, mu, cov):
    """
    Multivariate Normal logpdf
    
    Arguments:
        np.ndarray x:   value
        np.ndarray mu:  mean vector
        np.ndarray cov: covariance matrix
    
    Returns:
        double: MultivariateNormal(m,s).logpdf(x)
    """
    inv_cov  = inv_jit(cov)
    exponent = -0.5*scalar_product(x-mu, inv_cov, len(mu))
    lognorm  = 0.5*len(mu)*LOG2PI+0.5*logdet_jit(cov)
    return -lognorm+exponent

@njit
def log_norm_int(x, mu, cov_1, inv_cov_1, cov_2):
    """
    Multivariate Normal logpdf (tailored to this problem!)
    See https://arxiv.org/pdf/1811.04751v1.pdf
    
    Arguments:
        np.ndarray x:         value
        np.ndarray mu:        mean vector
        np.ndarray cov_1:     1st covariance matrix
        np.ndarray inv_cov_1: inverse of 1st covariance matrix
        np.ndarray cov_2:     2nd covariance matrix
    
    Returns:
        double: MultivariateNormal(m,s).logpdf(x)
    """
    inv_cov_2  = inv_jit(cov_2)
    inv_cov    = inv_cov_1@inv_jit(inv_cov_1+inv_cov_2)@inv_cov_2
    exponent   = -0.5*scalar_product(x-mu, inv_cov, len(mu))
    lognorm    = 0.5*len(mu)*LOG2PI+0.5*(logdet_jit(cov_1) + logdet_jit(cov_2) + logdet_jit(inv_cov_1+inv_cov_2))
    return -lognorm+exponent

#------------#
# 1D methods #
#------------#

@njit
def eval_mix_1d(mu, sigma, means, covs):
    """
    Computes N(mu_k| mu, (sigma_k^2+sigma^2) for all the components of a mixture (for predictive likelihood, 1D).
    
    Arguments:
        np.ndarray mu:    temptative mean of the parent mixture component
        np.ndarray sigma: temptative variance of the parent mixture component
        np.ndarray means: means of the event mixture components
        np.ndarray vars:  variances of the event mixture components
    
    Returns:
        np.ndarray: probability for each event mixture components
    """
    return np.array([log_norm_1d(means[i,0], mu, sigma+covs[i,0,0]) for i in prange(len(means))])

@njit
def evaluate_mixture_MC_draws_1d(mu, sigma, means, vars, w):
    """
    Computes N(mu_k| mu, (sigma_k^2+sigma^2) for a set of MC draws for mu and sigma.
    
    Arguments:
        np.ndarray mu:    MC draws for the mean of the parent mixture component
        np.ndarray sigma: MC draws for the variance of the parent mixture component
        np.ndarray means: means of the event mixture components
        np.ndarray vars:  variances of the event mixture components
        np.ndarray w:     component weights
    
    Returns:
        np.ndarray: probability for each MC draw
    """
    logP = np.zeros(len(mu), dtype = np.float64)
    for i in prange(len(mu)):
        logP[i] = logsumexp_jit_weighted(eval_mix_1d(mu[i], sigma[i], means, vars), b = w)
    return logP

#------------#
# ND methods #
#------------#

@njit
def eval_mix(mu, sigma, means, covs):
    """
    Computes N(mu_k| mu, (sigma_k^2+sigma^2) for all the components of a mixture (for predictive likelihood, ND).
    
    Arguments:
        np.ndarray mu:    temptative mean of the parent mixture component
        np.ndarray sigma: temptative covariance matrix of the parent mixture component
        np.ndarray means: means of the event mixture components
        np.ndarray covs:  covariance matrices of the event mixture components
    
    Returns:
        np.ndarray: probability for each event mixture components
    """
    inv_sigma = inv_jit(sigma)
    return np.array([log_norm_int(means[i], mu, sigma, inv_sigma, covs[i]) for i in prange(len(means))])

@njit
def evaluate_mixture_MC_draws(mu, sigma, means, covs, w):
    """
    Computes N(mu_k| mu, (sigma_k^2+sigma^2) for a set of MC draws for mu and sigma.
    
    Arguments:
        np.ndarray mu:    MC draws for the mean vector of the parent mixture component
        np.ndarray sigma: MC draws for the covariance matrix of the parent mixture component
        np.ndarray means: means of the event mixture components
        np.ndarray covs:  covariance matrices of the event mixture components
        np.ndarray w:     component weights
    
    Returns:
        np.ndarray: probability for each MC draw
    """
    logP = np.zeros(len(mu), dtype = np.float64)
    for i in prange(len(mu)):
        logP[i] = logsumexp_jit_weighted(eval_mix(mu[i], sigma[i], means, covs), b = w)
    return logP
