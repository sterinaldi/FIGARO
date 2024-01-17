import numpy as np
from numba import njit
from scipy.special import logsumexp
from figaro.exceptions import FIGAROException
from figaro._likelihood import log_norm
from figaro._numba_functions import inv_jit
from figaro.decorators import probit

@njit
def _cond_mean_cov(vals, mu1, mu2, s11, s22, s12):
    """
    Compute mean and covariance for conditional Normal distribution.
    See https://stats.stackexchange.com/questions/348941/general-conditional-distributions-for-multivariate-gaussian-mixtures
    
    Arguments:
        np.ndarray vals: values to condition on
        np.ndarray mu1:  mean subvector (NOT conditioned)
        np.ndarray mu2:  mean subvector (conditioned)
        np.ndarray s11:  covariance submatrix (NOT conditioned)
        np.ndarray s22:  covariance submatrix (conditioned)
        np.ndarray s12:  off-diagonal
    
    Returns:
        np.ndarray: mean
        np.ndarray: covariance
    """
    s22_inv = inv_jit(s22)
    mean = mu1 + s12@(s22_inv@(vals - mu2))
    cov  = s11 - s12@(s22_inv@s12.T)
    return mean, cov

def _marginalise(mix, axis = -1):
    """
    Marginalise out one or more dimensions from a FIGARO mixture.
    
    Arguments:
        figaro.mixture.mixture draws: mixture
        int or list of int axis:      axis to marginalise on
    
    Returns:
        figaro.mixture.mixture: the marginalised mixture
    """
    # Circular import
    from figaro.mixture import mixture
    ax     = np.atleast_1d(axis)
    dim    = mix.dim - len(ax)
    if dim < 1:
        raise FIGAROException("Cannot marginalise out all dimensions")
    means  = np.delete(mix.means, ax, axis = -1)
    covs   = np.delete(np.delete(mix.covs, ax, axis = -1), ax, axis = -2)
    bounds = np.delete(mix.bounds, ax, axis = 0)
    
    return mixture(means, covs, mix.w, bounds, dim, mix.n_cl, mix.n_pts, mix.alpha, probit = mix.probit)

def marginalise(draws, axis = -1):
    """
    Marginalise out one or more dimensions from a FIGARO draw.
    
    Arguments:
        figaro.mixture.mixture draws: mixture(s)
        int or list of int axis:      axis to marginalise on
    
    Returns:
        figaro.mixture.mixture: the marginalised mixture(s)
    """
    if axis == []:
        return draws
    if np.iterable(draws):
        return np.array([_marginalise(d, axis) for d in draws])
    else:
        return _marginalise(draws, axis)

@probit
def _condition(mix, vals, dims, norm = True, filter = True, tol = 1e-4):
    """
    Probability density conditioned on specific values of a subset of parameters.
    See:
     * https://stats.stackexchange.com/questions/348941/general-conditional-distributions-for-multivariate-gaussian-mixtures
     * https://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution
    
    Arguments:
        figaro.mixture.mixture mix: mixture
        iterable vals:              value(s) to condition on
        int or list of int dims:    dimension(s) associated with given vals (starting from 0)
        bool norm:                  normalize the distribution
        bool filter:                  filter the components with weight < tol
        double tol:                   tolerance on the sum of the weights
    
    Returns:
        figaro.mixture.mixture: the conditioned mixture(s)
    """
    # Circular import
    from figaro.mixture import mixture
    ax   = np.atleast_1d(dims)
    vals = vals[ax]
    dim  = mix.dim - len(ax)
    idx  = np.array([i in ax for i in range(mix.dim)])
    if dim < 1:
        raise FIGAROException("Cannot condition on all dimensions")
    bounds      = np.delete(mix.bounds, ax, axis = 0)
    means       = np.zeros(shape = (mix.n_cl, dim))
    covs        = np.zeros(shape = (mix.n_cl, dim, dim))
    log_weights = np.zeros(shape = mix.n_cl)
    for i, (mu, cov) in enumerate(zip(mix.means, mix.covs)):
        # Subvectors and submatrices
        mu1 = mu[~idx]
        mu2 = mu[idx]
        s11 = cov[~idx,:][:,~idx]
        s12 = cov[idx,:][:,~idx]
        s22 = cov[idx,:][:,idx]
        # Parameters
        means[i], covs[i] = _cond_mean_cov(vals, mu1, mu2, s11, s22, s12.T)
        log_weights[i]    = log_norm(vals, mu2, s22)
    # Weights
    log_weights = mix.log_w + log_weights
    if norm:
        log_weights -= _marginalise(mix, axis = np.arange(mix.dim)[~idx])._logpdf_probit(vals)
    # Filter out components with negligible weights
    idx_filt = [True for _ in range(len(log_weights))]
    norm_const = 0.
    if filter:
        ww = np.exp(log_weights)
        idx = np.argsort(ww)
        m = np.where(np.cumsum(ww[idx]) > tol*np.sum(ww))[0].min()
        idx_filt = [i in idx[m:] for i in range(len(ww))]
        if norm:
            log_weights -= logsumexp(log_weights[idx_filt])
    return mixture(means[idx_filt], covs[idx_filt], np.exp(log_weights[idx_filt]), bounds, dim, len(log_weights[idx_filt]), mix.n_pts, mix.alpha, probit = mix.probit, log_w = log_weights[idx_filt])

def condition(draws, vals, dims, norm = True, filter = True, tol = 1e-4):
    """
    Probability density conditioned on specific values of a subset of parameters.
    
    Arguments:
        figaro.mixture.mixture draws: mixture(s)
        iterable vals:                value(s) to condition on
        int or list of int dims:      dimension(s) associated with given vals (starting from 0)
        bool norm:                    normalize the distribution
        bool filter:                  filter the components with weight < tol
        double tol:                   tolerance on the sum of the weights
    
    Returns:
        figaro.mixture.mixture: the conditioned mixture(s)
    """
    if np.iterable(draws):
        v       = np.mean(draws[0].bounds, axis = -1)
        v[dims] = vals
        return np.array([_condition(d, v, dims, norm, filter, tol) for d in draws])
    else:
        v       = np.mean(draws.bounds, axis = -1)
        v[dims] = vals
        return _condition(draws, v, dims, norm, filter, tol)
