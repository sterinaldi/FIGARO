import numpy as np
from numba import njit
from numba.extending import get_cython_function_address
import ctypes

@njit
def inv_jit(M):
  return np.linalg.inv(M)

@njit
def dot_jit(v1, v2):
  return np.dot(v1, v2)

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
    std      = np.sqrt(diag_jit(S))
    std_filt = std
    std_filt[std == 0.] = 1. # avoids NaNs due to (0/0)*0
    rho = divide_jit(S, outer_jit(std_filt,std_filt))
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

def _mvn_logpdf(x, means, covs, inv_covs, det_covs):
    """
    Compute the probability density function of multiple multivariate Gaussian distributions for multiple samples.
    
    Arguments:
    np.ndarray samples:  2D array where each row represents a sample point (shape: [n_samples, n_dimensions]).
    np.ndarray means:    2D array where each row represents a mean vector for a distribution (shape: [n_distributions, n_dimensions]).
    np.ndarray covs:     3D array where each subarray is a covariance matrix for a distribution (shape: [n_distributions, n_dimensions, n_dimensions]).
    np.ndarray inv_covs: 3D array where each subarray is an inverse covariance matrix for a distribution (shape: [n_distributions, n_dimensions, n_dimensions]).
    np.ndarray det_covs: 1D array with the determinants of each covariance matrix.

    
    Returns:
        np.ndarray: 2D array of log_PDF values where each row corresponds to a sample and each column to a distribution.
    """
    n_distributions, k = means.shape # number of distributions and dimensions
    n_samples          = x.shape[0]  # number of samples
    # Normalization constants for each distribution
    norm_consts = 1 / ((2 * np.pi) ** (k / 2) * np.sqrt(det_covs))  # Shape: (n_distributions,)
    # Center each sample relative to each mean (broadcasted subtraction)
    centered_samples = x[:, np.newaxis, :] - means[np.newaxis, :, :]  # Shape: (n_samples, n_distributions, k)
    # Compute the exponent term using einsum for vectorized matrix multiplication
    # Result shape after einsum: (n_samples, n_distributions)
    exponent = -0.5 * np.einsum('sdi,dij,sdj->sd', centered_samples, inv_covs, centered_samples)
    # Calculate the PDF values for each sample against each distribution
    log_pdf_values = np.log(norm_consts) + (exponent)  # Shape: (n_samples, n_distributions)
    return log_pdf_values

def _mvn_pdf(x, means, covs, inv_covs, det_covs):
    return np.exp(_mvn_logpdf(x, means, covs, inv_covs, det_covs))
