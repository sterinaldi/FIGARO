import numpy as np
import sys
import dill

from collections import Counter
from pathlib import Path

from scipy.special import gammaln, logsumexp
from scipy.stats import multivariate_normal as mn
from scipy.stats import invwishart, norm, invgamma, dirichlet, gamma

from figaro.decorators import *
from figaro.transform import *
from figaro._numba_functions import *
from figaro._likelihood import evaluate_mixture_MC_draws, evaluate_mixture_MC_draws_1d, log_norm
from figaro.exceptions import except_hook, FIGAROException
from figaro.utils import get_priors
from figaro.marginal import _condition, _marginalise

from numba import njit, prange

sys.excepthook = except_hook

#-----------#
# Functions #
#-----------#
@njit
def _student_t(df, t, mu, sigma, dim):
    """
    Multivariate student-t pdf.
    As in http://gregorygundersen.com/blog/2020/01/20/multivariate-t/
    
    Arguments:
        float df:         degrees of freedom
        float t:          variable (2d array)
        np.ndarray mu:    mean (2d array)
        np.ndarray sigma: variance
        int dim:          number of dimensions
        
    Returns:
        float: student_t(df).logpdf(t)
    """
    vals, vecs = eigh_jit(sigma)
    logdet     = np.log(vals).sum()
    valsinv    = np.array([1./v for v in vals])
    U          = vecs * np.sqrt(valsinv)
    dev        = t - mu
    maha       = np.sum(np.dot(dev, U)**2, axis=-1)

    x = 0.5 * (df + dim)
    A = gammaln_jit(x)
    B = gammaln_jit(0.5 * df)
    C = dim/2. * np.log(df * np.pi)
    D = 0.5 * logdet
    E = -x * np.log1p((1./df) * maha)

    return (A - B - C - D + E)[0]

@njit
def _update_alpha(alpha, n, K, alpha0, burnin = 1000):
    """
    Update concentration parameter using a Metropolis-Hastings sampling scheme.
    
    Arguments:
        double alpha: Initial value for concentration parameter
        int n:        Number of samples
        int K:        Number of active clusters
        int burnin:   MH burnin
    
    Returns:
        double: new concentration parameter value
    """
    a_old = alpha
    n_draws = burnin+np.random.randint(100)
    for i in prange(n_draws):
        a_new = a_old + (np.random.random() - 0.5)*0.5
        if a_new > 0.:
            logP_old = gammaln_jit(a_old) - gammaln_jit(a_old + n) + (K+alpha0) * np.log(a_old) - a_old
            logP_new = gammaln_jit(a_new) - gammaln_jit(a_new + n) + (K+alpha0) * np.log(a_new) - a_new
            if logP_new - logP_old > np.log(np.random.random()):
                a_old = a_new
    return a_old

@njit
def _compute_t_pars(k, mu, nu, L, mean, S, N, dim):
    """
    Compute parameters for student-t distribution.
    
    Arguments:
        double k:        Normal std parameter (for NIW)
        np.ndarray mu:   Normal mean parameter (for NIW)
        int nu:          Inverse-Wishart df parameter (for NIW)
        np.ndarray L:    Inverse-Wishart scale matrix (for NIW)
        np.ndarray mean: samples mean
        np.ndarray S:    samples covariance
        int N:           number of samples
        int dim:         number of dimensions
    
    Returns:
        int:        degrees of fredom for student-t
        np.ndarray: scale matrix for student-t
        np.ndarray: mean for student-t
    """
    # Update hyperparameters
    k_n, mu_n, nu_n, L_n = _compute_hyperpars(k, mu, nu, L, mean, S, N)
    # Update t-parameters
    t_df    = nu_n - dim + 1
    t_shape = rescale_matrix(L_n, 1./((k_n+1.)/(k_n*t_df)))
    return t_df, t_shape, mu_n

@njit
def _compute_hyperpars(k, mu, nu, L, mean, S, N):
    """
    Update hyperparameters for Normal Inverse Gamma/Wishart (NIG/NIW).
    See https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    
    Arguments:
        double k:        Normal std parameter (for NIG/NIW)
        np.ndarray mu:   Normal mean parameter (for NIG/NIW)
        int nu:          Gamma df parameter (for NIG/NIW)
        np.ndarray L:    Gamma scale matrix (for NIG/NIW)
        np.ndarray mean: samples mean
        np.ndarray S:    samples covariance
        int N:           number of samples
    
    Returns:
        double:     updated Normal std parameter (for NIG/NIW)
        np.ndarray: updated Normal mean parameter (for NIG/NIW)
        int:        updated Gamma df parameter (for NIG/NIW)
        np.ndarray: updated Gamma scale matrix (for NIG/NIW)
    """
    k_n  = k + N
    mu_n = (mu*k + N*mean)/k_n
    nu_n = nu + N
    if N > 0:
        L_n = L + S + rescale_matrix(outer_jit((mean - mu).T, (mean - mu)), k_n/(k*N))
    else:
        L_n = L + S
    return k_n, mu_n, nu_n, L_n

@njit
def _compute_component_suffstats_add(x, mean, S, N, p_mu, p_k, p_nu, p_L):
    """
    Update mean, covariance, number of samples and maximum a posteriori for mean and covariance.
    
    Arguments:
        np.ndarray x:    sample to add
        np.ndarray mean: mean of samples already in the cluster
        np.ndarray cov:  covariance of samples already in the cluster
        int N:           number of samples already in the cluster
        np.ndarray p_mu: NIG Normal mean parameter
        double p_k:      NIG Normal std parameter
        int p_nu:        NIG Gamma df parameter
        np.ndarray p_L:  NIG Gamma scale matrix
    
    Returns:
        np.ndarray: updated mean
        np.ndarray: updated covariance
        int:        updated number of samples
        np.ndarray: mean (maximum a posteriori)
        np.ndarray: covariance (maximum a posteriori)
    """
    new_mean  = (mean*N+x)/(N+1)
    new_S     = (S + rescale_matrix(outer_jit(mean,mean), 1./N) + outer_jit(x,x)) - rescale_matrix(outer_jit(new_mean, new_mean), 1./(N+1))
    new_N     = N+1
    new_mu    = ((p_mu*p_k + new_N*new_mean)/(p_k + new_N))[0]
    new_sigma = rescale_matrix(p_L + new_S + rescale_matrix(outer_jit((new_mean - p_mu).T, (new_mean - p_mu)), (p_k + new_N)/(p_k*new_N)), (p_nu + new_N - x.shape[-1] - 1))
    
    return new_mean, new_S, new_N, new_mu, new_sigma

@njit
def _compute_component_suffstats_remove(x, mean, S, N, p_mu, p_k, p_nu, p_L):
    """
    Update mean, covariance, number of samples and maximum a posteriori for mean and covariance.
    
    Arguments:
        np.ndarray x:    sample to remove
        np.ndarray mean: mean of samples already in the cluster
        np.ndarray cov:  covariance of samples already in the cluster
        int N:           number of samples already in the cluster
        np.ndarray p_mu: NIG Normal mean parameter
        double p_k:      NIG Normal std parameter
        int p_nu:        NIG Gamma df parameter
        np.ndarray p_L:  NIG Gamma scale matrix
    
    Returns:
        np.ndarray: updated mean
        np.ndarray: updated covariance
        int:        updated number of samples
        np.ndarray: mean (maximum a posteriori)
        np.ndarray: covariance (maximum a posteriori)
    """
    if N > 1:
        new_mean  = (mean*N-x)/(N-1)
        if N > 2:
            new_S = (S + rescale_matrix(outer_jit(mean,mean), 1./N) - outer_jit(x,x)) - rescale_matrix(outer_jit(new_mean, new_mean), 1./(N-1))
        else:
            new_S = np.zeros(S.shape)
    else:
        new_mean  = np.zeros(mean.shape)
        new_S     = np.zeros(S.shape)
    new_N     = N-1
    new_mu    = ((p_mu*p_k + new_N*new_mean)/(p_k + new_N))[0]
    if N > 2:
        new_sigma = rescale_matrix(p_L + new_S + rescale_matrix(outer_jit((new_mean - p_mu).T, (new_mean - p_mu)), (p_k + new_N)/(p_k*new_N)), (p_nu + new_N - x.shape[-1] - 1))
    else:
        new_sigma = rescale_matrix(p_L, (p_nu + new_N - x.shape[-1] - 1))
    return new_mean, new_S, new_N, new_mu, new_sigma
    
#-------------------#
# Auxiliary classes #
#-------------------#

class _prior:
    """
    Class to store the NIW prior parameters
    See https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf, sec. 9
    
    Arguments:
        double k:        Normal std parameter
        np.ndarray L:    Wishart scale matrix
        int nu:          Wishart df parameter
        np.ndarray mu:   Normal mean parameter
    
    Returns:
        prior: instance of prior class
    """
    def __init__(self, k, L, nu, mu):
         self.k   = k
         self.nu  = np.max([nu, mu.shape[-1]+2])
         self.L   = rescale_matrix(L, 1./(self.nu-mu.shape[-1]-1))
         self.mu  = mu

class _component:
    """
    Class to store the relevant informations for each component in the mixture.
    
    Arguments:
        np.ndarray x: sample added to the new component
        prior prior:  instance of the prior class with NIG/NIW prior parameters
    
    Returns:
        component: instance of component class
    """
    def __init__(self, x, prior, N = 1):
        self.N     = N
        self.mean  = x
        self.S     = np.identity(x.shape[-1])*0.
        self.mu    = np.atleast_2d((prior.mu*prior.k + self.N*self.mean)/(prior.k + self.N)).astype(np.float64)[0]
        self.sigma = rescale_matrix(prior.L, (prior.nu - x.shape[-1] - 1))

class _component_h:
    """
    Class to store the relevant informations for each component in the mixture.
    To be used in hierarchical inference.
    
    Arguments:
        np.ndarray x:  event added to the new component
        int dim:       number of dimensions
        prior prior:   instance of the prior class with NIG/NIW prior parameters
        double logL_D: logLikelihood denominator
    
    Returns:
        component_h: instance of component_h class
    """
    def __init__(self, x, dim, prior, logL_D, mu_MC, sigma_MC):
        self.dim    = dim
        self.N      = 1
        self.logL_D = logL_D
        
        log_norm_D = logsumexp_jit(logL_D)
        
        idx        = np.random.choice(len(mu_MC), p = np.exp(logL_D - log_norm_D))
        self.mu    = np.copy(mu_MC[idx])
        self.sigma = np.copy(sigma_MC[idx])
        if dim == 1:
            self.mu = np.atleast_2d(self.mu).T
            self.sigma = np.atleast_2d(self.sigma).T
            
class density:
    """
    Class to initialise a common set of methods for mixture models. Not to be used.
    """
    def __init__(self):
        pass
        
    def __call__(self, x):
        return self.pdf(x)

    def pdf(self, x):
        if self.n_cl == 0:
            raise FIGAROException("You are trying to evaluate an empty mixture.\n If you are using the density_from_samples() method, you may want to evaluate the output of that method.")
        if len(np.shape(x)) < 2:
            if self.dim == 1:
                x = np.atleast_2d(x).T
            else:
                x = np.atleast_2d(x)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            p = np.nan_to_num(self._pdf(x), nan = 0.)
        return p

    def logpdf(self, x):
        if self.n_cl == 0:
            raise FIGAROException("You are trying to evaluate an empty mixture.\n If you are using the density_from_samples() method, you may want to evaluate the output of that method.")
        if len(np.shape(x)) < 2:
            if self.dim == 1:
                x = np.atleast_2d(x).T
            else:
                x = np.atleast_2d(x)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            logp = np.nan_to_num(self._logpdf(x), nan = -np.inf)
        return logp

    @probit
    def _pdf(self, x):
        """
        Evaluate mixture at point(s) x
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            np.ndarray: mixture.pdf(x)
        """
        return self._pdf_probit(x) * np.exp(-probit_logJ(x, self.bounds, self.probit))

    @probit
    def _logpdf(self, x):
        """
        Evaluate log mixture at point(s) x
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            np.ndarray: mixture.logpdf(x)
        """
        return self._logpdf_probit(x) - probit_logJ(x, self.bounds, self.probit)

    def fast_pdf(self, x):
        """
        Fast pdf evaluation using FIGARO implementation of log_norm (JIT) rather than Numpy's.
        WARNING: it is meant to be used with MCMC samplers, therefore accepts only one point at a time.
        
        Arguments:
            np.ndarray x: point to evaluate the mixture at
        
        Returns:
            np.ndarray: mixture.pdf(x)
        """
        x = np.atleast_1d(x)
        if x.shape == (1,self.dim):
            return self._fast_pdf(x[0])
        elif x.shape == (self.dim,):
            return self._fast_pdf(x)
        else:
            raise FIGAROException("Please provide one point at a time.")

    def fast_logpdf(self, x):
        """
        Fast logpdf evaluation using FIGARO implementation of log_norm (JIT) rather than Numpy's.
        WARNING: it is meant to be used with MCMC samplers, therefore accepts only one point at a time.
        
        Arguments:
            np.ndarray x: point to evaluate the mixture at
        
        Returns:
            np.ndarray: mixture.pdf(x)
        """
        x = np.atleast_1d(x)
        if x.shape == (1,self.dim):
            return self._fast_logpdf(x[0])
        elif x.shape == (self.dim,):
            return self._fast_logpdf(x)
        else:
            raise FIGAROException("Please provide one point at a time.")

    @probit
    def _fast_pdf(self, x):
        """
        Evaluate mixture at point x
        
        Arguments:
            np.ndarray x: point to evaluate the mixture at
        
        Returns:
            np.ndarray: mixture.pdf(x)
        """
        return self._fast_pdf_probit(x) * np.exp(-probit_logJ(x, self.bounds, self.probit))

    @probit
    def _fast_logpdf(self, x):
        """
        Evaluate log mixture at point x
        
        Arguments:
            np.ndarray x: point to evaluate the mixture at
        
        Returns:
            np.ndarray: mixture.logpdf(x)
        """
        return self._fast_logpdf_probit(x) - probit_logJ(x, self.bounds, self.probit)

    def _fast_pdf_probit(self, x):
        """
        Evaluate mixture at point x in probit space
        
        Arguments:
            np.ndarray x: point to evaluate the mixture at (in probit space)
        
        Returns:
            np.ndarray: mixture.pdf(x)
        """
        return np.sum(np.array([w*np.exp(log_norm(x[0], mean, cov)) for mean, cov, w in zip(self.means, self.covs, self.w)]), axis = 0)

    def _fast_logpdf_probit(self, x):
        """
        Evaluate log mixture at point x in probit space
        
        Arguments:
            np.ndarray x: point to evaluate the mixture at (in probit space)
        
        Returns:
            np.ndarray: mixture.logpdf(x)
        """
        return logsumexp(np.array([w + log_norm(x[0], mean, cov) for mean, cov, w in zip(self.means, self.covs, self.log_w)]), axis = 0)

    @probit
    def _pdf_no_jacobian(self, x):
        """
        Evaluate mixture at point(s) x without jacobian
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            np.ndarray: mixture.pdf(x)
        """
        return self._pdf_probit(x)

    def _pdf_probit(self, x):
        """
        Evaluate mixture at point(s) x in probit space
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at (in probit space)
        
        Returns:
            np.ndarray: mixture.pdf(x)
        """
        return np.sum(np.array([w*mn(mean, cov, allow_singular = True).pdf(x) for mean, cov, w in zip(self.means, self.covs, self.w)]), axis = 0)
    
    @probit
    def _pdf_array(self, x):
        """
        Evaluate every mixture component at point(s) x.
        
        Arguments:
            np.ndarray x: point(s) to evaluate the components at
        
        Returns:
            np.ndarray: component.pdf(x) for each mixture component
        """
        return _pdf_array_probit(x) * np.exp(-probit_logJ(x, self.bounds, self.probit))

    def _pdf_array_probit(self, x):
        """
        Evaluate every mixture component at point(s) x.
        
        Arguments:
            np.ndarray x: point(s) to evaluate the components at (in probit space)
        
        Returns:
            np.ndarray: component.pdf(x) for each mixture component
        """
        return np.array([w*mn(mean, cov, allow_singular = True).pdf(x) for mean, cov, w in zip(self.means, self.covs, self.w)])

    @probit
    def _fast_pdf_array(self, x):
        """
        Evaluate every mixture component at point(s) x.
        
        Arguments:
            np.ndarray x: point(s) to evaluate the components at
        
        Returns:
            np.ndarray: component.pdf(x) for each mixture component
        """
        return _fast_pdf_array_probit(x) * np.exp(-probit_logJ(x, self.bounds, self.probit))

    def _fast_pdf_array_probit(self, x):
        """
        Evaluate every mixture component at point(s) x.
        
        Arguments:
            np.ndarray x: point(s) to evaluate the components at (in probit space)
        
        Returns:
            np.ndarray: component.pdf(x) for each mixture component
        """
        return np.array([w*np.exp(log_norm(x[0], mean, cov)) for mean, cov, w in zip(self.means, self.covs, self.w)])

    @probit
    def _logpdf_no_jacobian(self, x):
        """
        Evaluate log mixture at point(s) x without jacobian
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            np.ndarray: mixture.logpdf(x)
        """
        return self._logpdf_probit(x)

    def _logpdf_probit(self, x):
        """
        Evaluate log mixture at point(s) x in probit space
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at (in probit space)
        
        Returns:
            np.ndarray: mixture.logpdf(x)
        """
        return logsumexp(np.array([w + mn(mean, cov, allow_singular = True).logpdf(x) for mean, cov, w in zip(self.means, self.covs, self.log_w)]), axis = 0)

    def cdf(self, x):
        if self.dim > 1:
            raise FIGAROException("cdf is provided only for 1-dimensional distributions")
        if len(np.shape(x)) < 2:
            x = np.atleast_2d(x).T
        return self._cdf(x)

    def logcdf(self, x):
        if self.dim > 1:
            raise FIGAROException("cdf is provided only for 1-dimensional distributions")
        if len(np.shape(x)) < 2:
            x = np.atleast_2d(x).T
        return self._logcdf(x)

    @probit
    def _cdf(self, x):
        """
        Evaluate mixture cdf at point(s) x
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            np.ndarray: mixture.cdf(x)
        """
        return np.sum(np.array([w*norm(mean[0], cov[0,0]).cdf(x) for mean, cov, w in zip(self.means, np.sqrt(self.covs), self.w)]), axis = 0)

    @probit
    def _logcdf(self, x):
        """
        Evaluate mixture log cdf at point(s) x
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            np.ndarray: mixture.logcdf(x)
        """
        return logsumexp(np.array([w + norm(mean[0], cov[0,0]).logcdf(x) for mean, cov, w in zip(self.means, np.sqrt(self.covs), self.log_w)]), axis = 0)

    @from_probit
    def rvs(self, size = 1):
        """
        Draw samples from mixture
        
        Arguments:
            int size: number of samples to draw
        
        Returns:
            np.ndarray: samples
        """
        if self.n_cl == 0:
            raise FIGAROException("You are trying to draw samples from an empty mixture.\n If you are using the density_from_samples() method, you may want to draw samples from the output of that method.")
        return self._rvs_probit(size)
        
    def _rvs_probit(self, size = 1):
        """
        Draw samples from mixture in probit space
        
        Arguments:
            int size: number of samples to draw
        
        Returns:
            np.ndarray: samples in probit space
        """
        idx = np.random.choice(np.arange(self.n_cl), p = self.w, size = size)
        ctr = Counter(idx)
        if self.dim > 1:
            samples = np.empty(shape = (1,self.dim))
            for i, n in zip(ctr.keys(), ctr.values()):
                samples = np.concatenate((samples, np.atleast_2d(mn(self.means[i], self.covs[i], allow_singular = True).rvs(size = n))))
        else:
            samples = np.array([np.zeros(1)])
            for i, n in zip(ctr.keys(), ctr.values()):
                samples = np.concatenate((samples, np.atleast_2d(mn(self.means[i], self.covs[i], allow_singular = True).rvs(size = n)).T))
        return np.array(samples[1:])
    
    def gradient(self, x):
        """
        Gradient of the mixture.
        
        Arguments:
            np.ndarray x: point to evaluate the gradient at
        
        Returns:
            np.ndarray: gradient
        """
        if self.n_cl == 0:
            raise FIGAROException("You are trying to evaluate an empty mixture.\n If you are using the density_from_samples() method, you may want to evaluate the output of that method.")
        if len(np.shape(x)) < 2:
            if self.dim == 1:
                x = np.atleast_2d(x).T
            else:
                x = np.atleast_2d(x)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            g = np.nan_to_num([self._gradient(xi) for xi in x], nan = 0.)
        return g
    
    def log_gradient(self, x):
        """
        Logarithmic gradient of the mixture.
        
        Arguments:
            np.ndarray x: point to evaluate the gradient at
        
        Returns:
            np.ndarray: logarithmic gradient
        """
        if self.n_cl == 0:
            raise FIGAROException("You are trying to evaluate an empty mixture.\n If you are using the density_from_samples() method, you may want to evaluate the output of that method.")
        if len(np.shape(x)) < 2:
            if self.dim == 1:
                x = np.atleast_2d(x).T
            else:
                x = np.atleast_2d(x)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            g = np.nan_to_num([self._log_gradient(xi) for xi in x], nan = 0.)
        return g
    
    def _gradient(self, x):
        """
        Gradient of the mixture.
        
        Arguments:
            np.ndarray x: point to evaluate the gradient at
        
        Returns:
            np.ndarray: gradient
        """
        return self._fast_pdf(x)*self._log_gradient(x)
    
    @probit
    def _log_gradient(self, x):
        """
        Logarithmic gradient of the mixture.
        
        Arguments:
            np.ndarray x: point to evaluate the gradient at
        
        Returns:
            np.ndarray: logarithmic gradient
        """
        p = self._pdf_array_probit(x)
        B = np.array([-np.dot(inv_jit(sigma),(x - mu)) for mu, sigma in zip(self.means, self.covs)])
        try:
            return np.average(B, weights = p, axis = 0) + log_gradient_inv_jacobian(x, self.bounds, self.probit)
        except ZeroDivisionError:
            return np.zeros(x.shape[-1])

class mixture(density):
    """
    Class to store a single draw from DPGMM/(H)DPGMM.
    Methods inherited from density class.
    
    Arguments:
        iterable means:    component means
        iterable covs:     component covariances
        np.ndarray w:      component weights
        np.ndarray bounds: bounds of probit transformation
        int dim:           number of dimensions
        int n_cl:          number of clusters in the mixture
        int n_pts:         number of points used to infer the mixture
        double alpha:      concentration parameter
        bool probit:       whether to use the probit transformation or not
    
    Returns:
        mixture: instance of mixture class
    """
    def __init__(self, means, covs, w, bounds, dim, n_cl, n_pts, alpha = 1., probit = True, log_w = None):
        self.means      = means
        self.covs       = covs
        self.components = [mn(mean, cov, allow_singular = True) for mean, cov in zip(self.means, self.covs)]
        if log_w is None:
            self.w      = w
            self.log_w  = np.log(w)
        else:
            self.log_w  = log_w
            self.w      = np.exp(log_w)
        self.bounds = bounds
        self.dim    = dim
        self.n_cl   = n_cl
        self.n_pts  = n_pts
        self.probit = probit
        self.alpha  = alpha
    
    def marginalise(self, axis = -1):
        """
        Marginalise out one or more dimensions from the mixture.
        
        Arguments:
            int or list of int axis: axis to marginalise on. Default: last
        
        Returns:
            figaro.mixture.mixture: marginalised mixture
        """
        return _marginalise(self, axis)
    
    def condition(self, vals, dims, norm = True, filter = True, tol = 1e-3):
        """
        Mixture conditioned on specific values of a subset of parameters.
        
        Arguments:
            iterable vals:           value(s) to condition on
            int or list of int dims: dimension(s) associated with given vals (starting from 0)
            bool norm:               whether to normalize the distribution or not
            bool filter:             filter the components with weight < tol
            double tol:              tolerance on the sum of the weights
        
        Returns:
            figaro.mixture.mixture: conditioned mixture
        """
        v       = np.mean(self.bounds, axis = -1)
        v[dims] = vals
        return _condition(self, v, dims, norm, filter, tol = 1e-3)

    def _logpdf_probit(self, x):
        """
        Evaluate log mixture at point(s) x in probit space
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at (in probit space)
        
        Returns:
            np.ndarray: mixture.logpdf(x)
        """
        return logsumexp(np.array([w+comp.logpdf(x) for w, comp in zip(self.log_w, self.components)]), axis = 0)

    def _pdf_probit(self, x):
        """
        Evaluate mixture at point(s) x in probit space
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at (in probit space)
        
        Returns:
            np.ndarray: mixture.pdf(x)
        """
        return np.sum(np.array([w*comp.pdf(x) for w, comp in zip(self.w, self.components)]), axis = 0)
        
#-------------------#
# Inference classes #
#-------------------#

class DPGMM(density):
    """
    Class to infer a distribution given a set of samples.
    
    Arguments:
        iterable bounds:     boundaries of the rectangle over which the distribution is defined. It should be in the format [[xmin, xmax],[ymin, ymax],...]
        iterable prior_pars: NIW prior parameters (k, L, nu, mu)
        double alpha0:       initial guess for concentration parameter
        bool probit:         whether to use the probit transformation or not
        int n_reassignments: number of reassignments. Default is not to reassign.
    
    Returns:
        DPGMM: instance of DPGMM class
    """
    def __init__(self, bounds,
                       prior_pars      = None,
                       alpha0          = 1.,
                       probit          = True,
                       n_reassignments = 0.,
                       ):
        self.probit = probit
        self.bounds = np.atleast_2d(bounds)
        self.dim    = len(self.bounds)
        if prior_pars is not None:
            self.prior = _prior(*prior_pars)
        else:
            self.prior = _prior(*get_priors(bounds = self.bounds, probit = self.probit))
        self.alpha           = alpha0
        self.alpha_0         = alpha0
        self.mixture         = []
        self.w               = []
        self.log_w           = []
        self.N_list          = []
        self.n_cl            = 0
        self.n_pts           = 0
        self.stored_pts      = {}
        self.assignations    = {}
        self.n_reassignments = int(n_reassignments)

    def __call__(self, x):
        return self.pdf(x)

    def initialise(self, prior_pars = None):
        """
        Initialise the mixture to initial conditions.
        
        Arguments:
            iterable prior_pars: NIW prior parameters (k, L, nu, mu). If None, old parameters are kept
        """
        self.alpha        = self.alpha_0
        self.mixture      = []
        self.w            = []
        self.log_w        = []
        self.N_list       = []
        self.n_cl         = 0
        self.n_pts        = 0
        self.stored_pts   = {}
        self.assignations = {}
        if prior_pars is not None:
            self.prior = _prior(*prior_pars)
        
    def _add_datapoint_to_component(self, x, ss):
        """
        Update component parameters after assigning a sample to a component
        
        Arguments:
            np.ndarray x: sample
            component ss: component to update
        
        Returns:
            component: updated component
        """
        new_mean, new_S, new_N, new_mu, new_sigma = _compute_component_suffstats_add(x, ss.mean, ss.S, ss.N, self.prior.mu, self.prior.k, self.prior.nu, self.prior.L)
        ss.mean  = new_mean
        ss.S     = new_S
        ss.N     = new_N
        ss.mu    = new_mu
        ss.sigma = new_sigma
        return ss

    def _remove_datapoint_from_component(self, x, ss):
        """
        Update component parameters after removing a sample from the component
        
        Arguments:
            np.ndarray x: sample
            component ss: component to update
        
        Returns:
            component: updated component
        """
        new_mean, new_S, new_N, new_mu, new_sigma = _compute_component_suffstats_remove(x, ss.mean, ss.S, ss.N, self.prior.mu, self.prior.k, self.prior.nu, self.prior.L)
        ss.mean  = new_mean
        ss.S     = new_S
        ss.N     = new_N
        ss.mu    = new_mu
        ss.sigma = new_sigma
        return ss

    def _log_predictive_likelihood(self, x, ss):
        """
        Compute log likelihood of drawing sample x from component ss given the samples that are already assigned to that component.
        
        Arguments:
            np.ndarray x: sample
            component ss: component to update
        
        Returns:
            double: log Likelihood
        """
        if ss is None:
            ss = _component(self.prior.mu, prior = self.prior, N = 0)
        t_df, t_shape, mu_n = _compute_t_pars(self.prior.k, self.prior.mu, self.prior.nu, self.prior.L, ss.mean, ss.S, ss.N, self.dim)
        try:
            return _student_t(df = t_df, t = x, mu = mu_n, sigma = t_shape, dim = self.dim)
        except np.linalg.LinAlgError:
            return -np.inf

    def _cluster_assignment_distribution(self, x):
        """
        Compute the marginal distribution of cluster assignment for each cluster.
        
        Arguments:
            np.ndarray x: sample
        
        Returns:
            dict: p_i for each component
        """
        scores = np.zeros(self.n_cl+1)
        for i in range(self.n_cl+1):
            if i == 0:
                ss        = None
                scores[i] = np.log(self.alpha)
            else:
                ss        = self.mixture[i-1]
                with np.errstate(divide = 'ignore', invalid = 'ignore'):
                    scores[i] = np.log(ss.N)
            if np.isfinite(scores[i]):
                scores[i] += self._log_predictive_likelihood(x, ss)
            else:
                scores[i] = -np.inf
        norm = logsumexp_jit(scores)
        return np.exp(scores - norm)

    def _assign_to_cluster(self, x, pt_id = None):
        """
        Assign the new sample x to an existing cluster or to a new cluster according to the marginal distribution of cluster assignment.
        
        Arguments:
            np.ndarray x: sample
            int pt_id:    point id
        """
        if pt_id is None:
            pt_id = len(list(self.stored_pts.keys()))-1
        self.n_pts += 1
        scores = self._cluster_assignment_distribution(x)
        cid = np.random.choice(np.arange(-1,self.n_cl), p=scores)
        if cid == -1:
            self.mixture.append(_component(x, prior = self.prior))
            self.N_list.append(1.)
            self.n_cl += 1
            self.assignations[pt_id] = int(self.n_cl)-1
        else:
            self.mixture[int(cid)] = self._add_datapoint_to_component(x, self.mixture[int(cid)])
            self.N_list[int(cid)] += 1
            self.assignations[pt_id] = int(cid)
        # Update weights
        self.w = np.array(self.N_list)
        # Beware of empty clusters!
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            self.w     = self.w/self.w.sum()
            self.log_w = np.log(self.w)
        return
    
    def _remove_from_cluster(self, x, cid):
        """
        Remove the sample x from its assigned cluster.
        
        Arguments:
            np.ndarray x: sample
            int cid:      cluster id
        """
        self.mixture[int(cid)] = self._remove_datapoint_from_component(x, self.mixture[int(cid)])
        self.N_list[int(cid)] -= 1
        self.n_pts            -= 1
        # Update weights
        self.w = np.array(self.N_list)
        # Beware of empty clusters!
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            self.w = self.w/self.w.sum()
            self.log_w = np.log(self.w)
        return
    
    def density_from_samples(self, samples):
        """
        Reconstruct the probability density from a set of samples.
        
        Arguments:
            iterable samples: samples set
        
        Returns:
            mixture: the inferred mixture
        """
        np.random.shuffle(samples)
        samples = np.ascontiguousarray(samples)
        # Thermalise
        for s in samples:
            self.add_new_point(s)
        # Random Gibbs walk (if required)
        for id in np.random.choice(self.n_pts, size = self.n_reassignments, replace = True):
            self._reassign_point(id)
        d = self.build_mixture()
        self.initialise()
        return d
    
    @probit
    def add_new_point(self, x):
        """
        Update the probability density reconstruction adding a new sample
        
        Arguments:
            np.ndarray x: sample
        """
        self.stored_pts[len(list(self.stored_pts.keys()))] = np.atleast_2d(x)
        self._assign_to_cluster(np.atleast_2d(x))
        self.alpha = _update_alpha(self.alpha, self.n_pts, self.n_cl, self.alpha_0)
    
    def _reassign_point(self, id):
        """
        Update the probability density reconstruction reassigining an existing sample
        
        Arguments:
            id: sample id
        """
        x   = self.stored_pts[id]
        cid = self.assignations[id]
        self._remove_from_cluster(x, cid)
        self._assign_to_cluster(x, id)
        self.alpha = _update_alpha(self.alpha, self.n_pts, (np.array(self.N_list) > 0).sum(), self.alpha_0)

    def build_mixture(self):
        """
        Instances a mixture class representing the inferred distribution
        
        Returns:
            mixture: the inferred distribution
        """
        if self.n_cl == 0:
            raise FIGAROException("You are trying to build an empty mixture - perhaps you called the initialise() method. If you are using the density_from_samples() method, the inferred mixture is returned by that method as an instance of mixture class.")
        # Number of active clusters
        n_cl      = (np.array(self.N_list) > 0).sum()
        means     = np.zeros((n_cl, self.dim))
        variances = np.zeros((n_cl, self.dim, self.dim))
        i = 0
        for ss in self.mixture:
            if ss.N > 0:
                k_n, mu_n, nu_n, L_n = _compute_hyperpars(self.prior.k, self.prior.mu, self.prior.nu, self.prior.L, ss.mean, ss.S, ss.N)
                variances[i] = invwishart(df = nu_n, scale = L_n).rvs()
                means[i]     = mn(mean = mu_n[0], cov = rescale_matrix(variances[i], k_n), allow_singular = True).rvs()
                i += 1
        w = dirichlet(self.w[self.w > 0]*self.n_pts+self.alpha/self.n_cl).rvs()[0]
        return mixture(means, variances, w, self.bounds, self.dim, n_cl, self.n_pts, self.alpha, probit = self.probit)

    # Methods to overwrite density methods
    def _rvs_probit(self, size = 1):
        """
        Draw samples from mixture in probit space
        
        Arguments:
            int size: number of samples to draw
        
        Returns:
            np.ndarray: samples in probit space
        """
        idx = np.random.choice(np.arange(self.n_cl), p = self.w, size = size)
        ctr = Counter(idx)
        if self.dim > 1:
            samples = np.empty(shape = (1,self.dim))
            for i, n in zip(ctr.keys(), ctr.values()):
                samples = np.concatenate((samples, np.atleast_2d(mn(self.mixture[i].mu, self.mixture[i].sigma, allow_singular = True).rvs(size = n))))
        else:
            samples = np.array([np.zeros(1)])
            for i, n in zip(ctr.keys(), ctr.values()):
                samples = np.concatenate((samples, np.atleast_2d(mn(self.mixture[i].mu[0], self.mixture[i].sigma, allow_singular = True).rvs(size = n)).T))
        return samples[1:]

    def _pdf_probit(self, x):
        """
        Evaluate mixture at point(s) x in probit space
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at (in probit space)
        
        Returns:
            np.ndarray: mixture.pdf(x)
        """
        return np.sum(np.array([w*mn(comp.mu, comp.sigma, allow_singular = True).pdf(x) for comp, w in zip(self.mixture, self.w)]), axis = 0)

    def _logpdf_probit(self, x):
        """
        Evaluate log mixture at point(s) x in probit space
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at (in probit space)
        
        Returns:
            np.ndarray: mixture.logpdf(x)
        """
        return logsumexp(np.array([w + mn(comp.mu, comp.sigma, allow_singular = True).logpdf(x) for comp, w in zip(self.mixture, self.log_w)]), axis = 0)

    def _fast_pdf_probit(self, x):
        """
        Evaluate mixture at point x in probit space
        
        Arguments:
            np.ndarray x: point to evaluate the mixture at (in probit space)
        
        Returns:
            np.ndarray: mixture.pdf(x)
        """
        return np.sum(np.array([w*np.exp(log_norm(x[0], comp.mean, comp.cov)) for comp, w in zip(self.mixture, self.w)]), axis = 0)

    def _fast_logpdf_probit(self, x):
        """
        Evaluate log mixture at point x in probit space
        
        Arguments:
            np.ndarray x: point to evaluate the mixture at (in probit space)
        
        Returns:
            np.ndarray: mixture.logpdf(x)
        """
        return logsumexp(np.array([w + log_norm(x[0], comp.mean, comp.cov) for comp, w in zip(self.mixture, self.log_w)]), axis = 0)

class HDPGMM(DPGMM):
    """
    Class to infer a distribution given a set of observations (each being a set of samples).
    Child of DPGMM class
    
    Arguments:
        iterable bounds:     boundaries of the rectangle over which the distribution is defined. It should be in the format [[xmin, xmax],[ymin, ymax],...]
        iterable prior_pars: IW parameters
        double alpha0:       initial guess for concentration parameter
        double MC_draws:     number of MC draws for integral
        bool probit:         whether to use the probit transformation or not
        int n_reassignments: number of reassignments. Default is not to reassign.
    
    Returns:
        HDPGMM: instance of HDPGMM class
    """
    def __init__(self, bounds,
                       alpha0          = 1.,
                       prior_pars      = None,
                       MC_draws        = None,
                       probit          = True,
                       n_reassignments = 0.,
                       ):
        bounds   = np.atleast_2d(bounds)
        self.dim = len(bounds)
        super().__init__(bounds = bounds, alpha0 = alpha0, probit = probit, n_reassignments = n_reassignments)
        if prior_pars is not None:
            self.exp_sigma, self.a = prior_pars
        else:
            self.exp_sigma, self.a = get_priors(bounds = self.bounds, probit = self.probit, hierarchical = True)
        self.invgamma = invgamma(self.a)
        if MC_draws is None:
            self.MC_draws = int((self.dim+1)*1e3)
        else:
            self.MC_draws = int(MC_draws)
        self.evaluated_logL = {}
        # MC samples
        self._draw_MC_samples()
        
    def initialise(self):
        """
        Initialise the mixture to initial conditions
        """
        super().initialise()
        self.evaluated_logL = {}
        self._draw_MC_samples()
    
    def _draw_MC_samples(self):
        """
        Draws MC samples for mu and sigma
        """
        self.sigma_MC = np.array([self.invgamma.rvs(self.MC_draws)*(self.exp_sigma[i]**2*(self.a+1)) for i in range(self.dim)]).T
        self.mu_MC    = np.random.uniform(low = self.bounds[:,0], high = self.bounds[:,1], size = (self.MC_draws, self.dim))
        if self.probit:
            self.mu_MC = transform_to_probit(self.mu_MC, self.bounds)
        if self.dim == 1:
            self.sigma_MC = self.sigma_MC.flatten()
            self.mu_MC = self.mu_MC.flatten()
        else:
            rhos = invwishart(df = self.dim+2, scale = np.identity(self.dim)).rvs(size = self.MC_draws)
            rhos = np.array([r/outer_jit(np.sqrt(diag_jit(r)), np.sqrt(diag_jit(r))) for r in rhos])
            self.sigma_MC = np.array([r*_outerjit(s,s) for r, s in zip(rhos, np.sqrt(self.sigma_MC))])
            
    def add_new_point(self, ev):
        """
        Update the probability density reconstruction adding a new sample
        
        Arguments:
            iterable ev: set of single-event draws from a DPGMM inference
        """
        x = np.random.choice(ev)
        self.stored_pts[len(list(self.stored_pts.keys()))] = x
        self._assign_to_cluster(x)
        self.alpha = _update_alpha(self.alpha, self.n_pts, self.n_cl, self.alpha_0)

    def _cluster_assignment_distribution(self, x, logL_x = None):
        """
        Compute the marginal distribution of cluster assignment for each cluster.
        
        Arguments:
            figaro.mixture.mixture x: sample
            double logL_x:            evaluated log likelihood
        
        Returns:
            dict: p_i for each component
        """
        scores = np.zeros(self.n_cl+1)
        logL_N = np.zeros((self.n_cl+1, self.MC_draws))
        if logL_x is None:
            if self.dim == 1:
                logL_x = evaluate_mixture_MC_draws_1d(self.mu_MC, self.sigma_MC, x.means, x.covs, x.w)
            else:
                logL_x = evaluate_mixture_MC_draws(self.mu_MC, self.sigma_MC, x.means, x.covs, x.w)
        for i in range(self.n_cl+1):
            if i == 0:
                ss = None
                logL_D = np.zeros(self.MC_draws)
                scores[i] = np.log(self.alpha)
            else:
                ss        = self.mixture[i-1]
                logL_D    = ss.logL_D
                with np.errstate(divide = 'ignore', invalid = 'ignore'):
                    scores[i] = np.log(ss.N)
            if np.isfinite(scores[i]):
                scores[i] += logsumexp_jit(logL_D + logL_x) - logsumexp_jit(logL_D)
            else:
                scores[i] = -np.inf
            logL_N[i]  = logL_D + logL_x
        norm   = logsumexp_jit(scores)
        scores = np.exp(scores-norm)
        return scores, logL_N, logL_x

    def _assign_to_cluster(self, x, pt_id = None, logL_x = None):
        """
        Assign the new sample x to an existing cluster or to a new cluster according to the marginal distribution of cluster assignment.
        
        Arguments:
            np.ndarray x: sample
            int pt_id:    point id
            logL_x:       evaluated log likelihood
        """
        if pt_id is None:
            pt_id = len(list(self.stored_pts.keys()))-1
        self.n_pts += 1
        scores, logL_N, logL_x = self._cluster_assignment_distribution(x, logL_x)
        try:
            cid = np.random.choice(np.arange(-1, self.n_cl), p=scores)
        except ValueError:
            cid = -1
        if cid == -1:
            self.mixture.append(_component_h(x, self.dim, self.prior, logL_N[int(cid)+1], self.mu_MC, self.sigma_MC))
            self.N_list.append(1.)
            self.n_cl += 1
            self.assignations[pt_id] = int(self.n_cl)-1
        else:
            self.mixture[int(cid)] = self._add_datapoint_to_component(self.mixture[int(cid)], logL_N[int(cid)+1])
            self.N_list[int(cid)] += 1
            self.assignations[pt_id] = int(cid)
        self.evaluated_logL[pt_id] = logL_x
        # Update weights
        self.w = np.array(self.N_list)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            self.w = self.w/self.w.sum()
            self.log_w = np.log(self.w)
        return

    def _remove_from_cluster(self, x, cid, logL_x):
        """
        Remove the sample x from its assigned cluster.
        
        Arguments:
            np.ndarray x:  sample
            int cid:       cluster id
            double logL_x: log likelihood for the point
        """
        self.mixture[int(cid)] = self._remove_datapoint_from_component(self.mixture[int(cid)], self.mixture[int(cid)].logL_D - logL_x)
        self.N_list[int(cid)] -= 1
        self.n_pts            -= 1
        # Update weights
        self.w = np.array(self.N_list)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            self.w     = self.w/self.w.sum()
            self.log_w = np.log(self.w)
        return
    
    def _add_datapoint_to_component(self, ss, logL_D):
        """
        Update component parameters after assigning a sample to a component
        
        Arguments:
            component ss: component to update
            double logL_D: log Likelihood denominator
        
        Returns:
            component: updated component
        """
        ss.logL_D = logL_D
        log_norm_D = logsumexp_jit(logL_D)
        
        idx      = np.random.choice(self.MC_draws, p = np.exp(logL_D - log_norm_D))
        ss.mu    = np.copy(self.mu_MC[idx])
        ss.sigma = np.copy(self.sigma_MC[idx])
        if self.dim == 1:
            ss.mu = np.atleast_2d(ss.mu).T
            ss.sigma = np.atleast_2d(ss.sigma).T
        
        ss.N += 1
        return ss

    def _remove_datapoint_from_component(self, ss, logL_D):
        """
        Update component parameters after assigning a sample to a component
        
        Arguments:
            component ss: component to update
            double logL_D: log Likelihood denominator
        
        Returns:
            component: updated component
        """
        ss.logL_D = logL_D
        log_norm_D = logsumexp_jit(logL_D)
        
        idx      = np.random.choice(self.MC_draws, p = np.exp(logL_D - log_norm_D))
        ss.mu    = np.copy(self.mu_MC[idx])
        ss.sigma = np.copy(self.sigma_MC[idx])
        if self.dim == 1:
            ss.mu = np.atleast_2d(ss.mu).T
            ss.sigma = np.atleast_2d(ss.sigma).T
        
        ss.N -= 1
        return ss

    def build_mixture(self):
        """
        Instances a mixture class representing the inferred distribution
        
        Returns:
            mixture: the inferred distribution
        """
        if self.n_cl == 0:
            raise FIGAROException("You are trying to build an empty mixture - perhaps you called the initialise() method. If you are using the density_from_samples() method, the inferred mixture is returned by that method as an instance of mixture class.")
        idx = np.where(np.array(self.N_list) > 0)[0]
        return mixture(np.array([comp.mu[0] for comp in np.array(self.mixture)[idx]]), np.array([comp.sigma for comp in np.array(self.mixture)[idx]]), np.array(self.w)[idx], self.bounds, self.dim, (np.array(self.N_list) > 0).sum(), self.n_pts, self.alpha, probit = self.probit)

    def density_from_samples(self, events):
        """
        Reconstruct the probability density from a set of samples.
        
        Arguments:
            iterable samples: set of single-event draws from DPGMM
        
        Returns:
            mixture: the inferred mixture
        """
        np.random.shuffle(events)
        for ev in events:
            self.add_new_point(ev)
        # Random Gibbs walk (if required)
        for id in np.random.choice(self.n_pts, size = self.n_reassignments, replace = True):
            self._reassign_point(id)
        d = self.build_mixture()
        self.initialise()
        return d

    def _reassign_point(self, id):
        """
        Update the probability density reconstruction reassigining an existing sample
        
        Arguments:
            id: sample id
        """
        x      = self.stored_pts[id]
        cid    = self.assignations[id]
        logL_x = self.evaluated_logL[id]
        self._remove_from_cluster(x, cid, logL_x)
        self._assign_to_cluster(x, id, logL_x)
        self.alpha = _update_alpha(self.alpha, self.n_pts, (np.array(self.N_list) > 0).sum(), self.alpha_0)
