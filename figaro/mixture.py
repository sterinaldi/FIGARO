import numpy as np
import sys
import dill

from collections import Counter
from pathlib import Path

from scipy.special import gammaln, logsumexp
from scipy.stats import multivariate_normal as mn
from scipy.stats import invwishart

from figaro.decorators import *
from figaro.transform import *
from figaro.metropolis import sample_point, sample_point_1d, MC_predictive_1d, MC_predictive
from figaro.exceptions import except_hook

from numba import jit, njit, prange
from numba.extending import get_cython_function_address
import ctypes

#-----------#
# Utilities #
#-----------#

sys.excepthook = except_hook

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
functype = ctypes.CFUNCTYPE(_dble, _dble)
gammaln_float64 = functype(addr)

#-----------#
# Functions #
#-----------#

@njit
def numba_gammaln(x):
    return gammaln_float64(x)

@jit
def student_t(df, t, mu, sigma, dim):
    """
    Multivariate student-t pdf.
    As in http://gregorygundersen.com/blog/2020/01/20/multivariate-t/
    
    Arguments:
        :float df:         degrees of freedom
        :float t:          variable
        :np.ndarray mu:    mean
        :np.ndarray sigma: variance
        :int dim:          number of dimensions
        
    Returns:
        :float: student_t(df).logpdf(t)
    """
    vals, vecs = np.linalg.eigh(sigma)
    logdet     = np.log(vals).sum()
    valsinv    = np.array([1./v for v in vals])
    U          = vecs * np.sqrt(valsinv)
    dev        = t - mu
    maha       = np.square(np.dot(dev, U)).sum(axis=-1)

    x = 0.5 * (df + dim)
    A = numba_gammaln(x)
    B = numba_gammaln(0.5 * df)
    C = dim/2. * np.log(df * np.pi)
    D = 0.5 * logdet
    E = -x * np.log1p((1./df) * maha)

    return (A - B - C - D + E)[0]

@jit
def update_alpha(alpha, n, K, burnin = 1000):
    """
    Update concentration parameter using a Metropolis-Hastings sampling scheme.
    
    Arguments:
        :int n:      Number of samples
        :int K:      Number of active clusters
        :int burnin: MH burnin
    
    Returns:
        :double: new concentration parameter value
    """
    a_old = alpha
    n_draws = burnin+np.random.randint(100)
    for i in prange(n_draws):
        a_new = a_old + (np.random.random() - 0.5)
        if a_new > 0.:
            logP_old = numba_gammaln(a_old) - numba_gammaln(a_old + n) + K * np.log(a_old) - 1./a_old
            logP_new = numba_gammaln(a_new) - numba_gammaln(a_new + n) + K * np.log(a_new) - 1./a_new
            if logP_new - logP_old > np.log(np.random.random()):
                a_old = a_new
    return a_old

@jit
def compute_t_pars(k, mu, nu, L, mean, S, N, dim):
    """
    Compute parameters for student-t distribution.
    
    Arguments:
        :double k:        Normal std parameter (for NIG/NIW)
        :np.ndarray mu:   Normal mean parameter (for NIG/NIW)
        :int nu:          Gamma df parameter (for NIG/NIW)
        :np.ndarray L:    Gamma scale matrix (for NIG/NIW)
        :np.ndarray mean: samples mean
        :np.ndarray S:    samples covariance
        :int N:           number of samples
        :int dim:         number of dimensions
    
    Returns:
        :int:        degrees of fredom for student-t
        :np.ndarray: scale matrix for student-t
        :np.ndarray: mean for student-t
    """
    # Update hyperparameters
    k_n, mu_n, nu_n, L_n = compute_hyperpars(k, mu, nu, L, mean, S, N)
    # Update t-parameters
    t_df    = nu_n - dim + 1
    t_shape = L_n*(k_n+1)/(k_n*t_df)
    return t_df, t_shape, mu_n

@jit
def compute_hyperpars(k, mu, nu, L, mean, S, N):
    """
    Update hyperparameters for Normal Inverse Gamma/Wishart (NIG/NIW).
    See https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    
    Arguments:
        :double k:        Normal std parameter (for NIG/NIW)
        :np.ndarray mu:   Normal mean parameter (for NIG/NIW)
        :int nu:          Gamma df parameter (for NIG/NIW)
        :np.ndarray L:    Gamma scale matrix (for NIG/NIW)
        :np.ndarray mean: samples mean
        :np.ndarray S:    samples covariance
        :int N:           number of samples
    
    Returns:
        :double:     updated Normal std parameter (for NIG/NIW)
        :np.ndarray: updated Normal mean parameter (for NIG/NIW)
        :int:        updated Gamma df parameter (for NIG/NIW)
        :np.ndarray: updated Gamma scale matrix (for NIG/NIW)
    """
    k_n  = k + N
    mu_n = (mu*k + N*mean)/k_n
    nu_n = nu + N
    L_n  = L*k + S*N + k*N*((mean - mu).T@(mean - mu))/k_n
    return k_n, mu_n, nu_n, L_n

@jit
def compute_component_suffstats(x, mean, cov, N, p_mu, p_k, p_nu, p_L):
    """
    Update mean, covariance, number of samples and maximum a posteriori for mean and covariance.
    
    Arguments:
        :np.ndarray x:    sample to add
        :np.ndarray mean: mean of samples already in the cluster
        :np.ndarray cov:  covariance of samples already in the cluster
        :int N:           number of samples already in the cluster
        :np.ndarray p_mu: NIG Normal mean parameter
        :double p_k:      NIG Normal std parameter
        :int p_nu:        NIG Gamma df parameter
        :np.ndarray p_L:  NIG Gamma scale matrix
    
    Returns:
        :np.ndarray: updated mean
        :np.ndarray: updated covariance
        :int N:      updated number of samples
        :np.ndarray: mean (maximum a posteriori)
        :np.ndarray: covariance (maximum a posteriori)
    """
    new_mean  = (mean*N+x)/(N+1)
    new_cov   = (N*(cov + mean.T@mean) + x.T@x)/(N+1) - new_mean.T@new_mean
    new_N     = N+1
    new_mu    = ((p_mu*p_k + new_N*new_mean)/(p_k + new_N))[0]
    new_sigma = (p_L*p_k + new_cov*new_N + p_k*new_N*((new_mean - p_mu).T@(new_mean - p_mu))/(p_k + new_N))/(p_nu + new_N)
    
    return new_mean, new_cov, new_N, new_mu, new_sigma

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
    mean  = np.atleast_2d(x[:dim])
    corr  = np.identity(dim)/2.
    corr[np.triu_indices(dim, 1)] = x[2*dim:]
    corr  = corr + corr.T
    sigma = x[dim:2*dim]
    cov_mat = np.multiply(corr, np.outer(sigma, sigma))
    return mean, cov_mat

#-------------------#
# Auxiliary classes #
#-------------------#

class prior:
    """
    Class to store the NIG/NIW prior parameters
    See https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    
    Arguments:
        :double k:        Normal std parameter
        :np.ndarray mu:   Normal mean parameter
        :int nu:          Gamma df parameter
        :np.ndarray L:    Gamma scale matrix
    
    Returns:
        :prior: instance of prior class
    """
    def __init__(self, k, L, nu, mu):
        self.k = k
        self.L = L
        self.mu = mu
        self.nu = nu

class component:
    """
    Class to store the relevant informations for each component in the mixture.
    
    Arguments:
        :np.ndarray x: sample added to the new component
        :prior prior:  instance of the prior class with NIG/NIW prior parameters
    
    Returns:
        :component: instance of component class
    """
    def __init__(self, x, prior):
        self.N     = 1
        self.mean  = x
        self.cov   = np.identity(x.shape[-1])*0.
        self.mu    = np.atleast_2d((prior.mu*prior.k + self.N*self.mean)/(prior.k + self.N)).astype(np.float64)[0]
        self.sigma = np.identity(x.shape[-1]).astype(np.float64)*prior.L

class component_h:
    """
    Class to store the relevant informations for each component in the mixture.
    To be used in hierarchical inference.
    
    Arguments:
        :np.ndarray x:  event added to the new component
        :int dim:       number of dimensions
        :prior prior:   instance of the prior class with NIG/NIW prior parameters
        :double logL_D: logLikelihood denominator
    
    Returns:
        :component_h: instance of component_h class
    """
    def __init__(self, x, dim, prior, logL_D):
        self.dim    = dim
        self.N      = 1
        self.events = [x]
        self.means  = [x.means]
        self.covs   = [x.covs]
        self.log_w  = [x.log_w]
        self.logL_D = logL_D
        
        if self.dim == 1:
            sample = sample_point_1d(self.means, self.covs, self.log_w, a = prior.nu+1, b = prior.L[0,0])
        else:
            sample = sample_point(self.means, self.covs, self.log_w, self.dim, a = prior.nu, b = prior.L)
        self.mu, self.sigma = build_mean_cov(sample, self.dim)
    
class mixture:
    """
    Class to store a single draw from DPGMM/(H)DPGMM.
    
    Arguments:
        :iterable means:    component means
        :iterable covs:     component covariances
        :np.ndarray w:      component weights
        :np.ndarray bounds: bounds of probit transformation
        :int dim:           number of dimensions
        :int n_cl:          number of clusters in the mixture
        :int n_draws:       number of MC draws for normalisation constant estimate
        :bool hier_flag:    flag for hierarchical mixture (needed to fix an issue with means)
    
    Returns:
        :mixture: instance of mixture class
    """
    def __init__(self, means, covs, w, bounds, dim, n_cl, n_pts, n_draws = 1000, hier_flag = False):
        if dim > 1 and hier_flag:
            self.means = np.array([m[0] for m in means])
        else:
            self.means = means
        self.covs     = covs
        self.w        = w
        self.log_w    = np.log(w)
        self.bounds   = bounds
        self.dim      = dim
        self.n_cl     = n_cl
        self.n_pts    = n_pts
        self.norm     = 1.
        self.norm     = self._compute_norm_const(n_draws)
        self.log_norm = np.log(self.norm)
    
    def _compute_norm_const(self, n_draws = 1000):
        """
        Estimate normalisation constant via MC integration
        
        Arguments:
            :int n_draws: number of MC draws
        
        Returns:
            :double: normalisation constant
        """
        p_ss     = self.sample_from_dpgmm(n_draws)
        min_vals = np.atleast_1d(p_ss.min(axis = 0))
        max_vals = np.atleast_1d(p_ss.max(axis = 0))
        volume   = np.prod(np.diff(np.array([min_vals, max_vals]).T))
        ss       = np.random.uniform(min_vals, max_vals, size = (n_draws, self.dim))
        return self.evaluate_mixture(np.atleast_2d(ss)).sum()*volume/n_draws
        
    @probit
    def evaluate_mixture(self, x):
        """
        Evaluate mixture at point(s) x
        
        Arguments:
            :np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            :np.ndarray: mixture.pdf(x)
        """
        p = np.sum(np.array([w*mn(mean, cov).pdf(x)/self.norm for mean, cov, w in zip(self.means, self.covs, self.w)]), axis = 0)
        return p * np.exp(-probit_logJ(x, self.bounds))

    @probit
    def evaluate_log_mixture(self, x):
        """
        Evaluate log mixture at point(s) x
        
        Arguments:
            :np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            :np.ndarray: mixture.logpdf(x)
        """
        p = logsumexp(np.array([w + mn(mean, cov).logpdf(x) - self.log_norm for mean, cov, w in zip(self.means, self.covs, self.log_w)]), axis = 0)
        return p - probit_logJ(x, self.bounds)
        
    def _evaluate_mixture_in_probit(self, x):
        """
        Evaluate mixture at point(s) x in probit space
        
        Arguments:
            :np.ndarray x: point(s) to evaluate the mixture at (in probit space)
        
        Returns:
            :np.ndarray: mixture.pdf(x)
        """
        p = np.sum(np.array([w*mn(mean, cov).pdf(x) for mean, cov, w in zip(self.means, self.covs, self.w)]), axis = 0)
        return p

    def _evaluate_log_mixture_in_probit(self, x):
        """
        Evaluate log mixture at point(s) x in probit space
        
        Arguments:
            :np.ndarray x: point(s) to evaluate the mixture at (in probit space)
        
        Returns:
            :np.ndarray: mixture.logpdf(x)
        """
        p = logsumexp(np.array([w + mn(mean, cov).logpdf(x) for mean, cov, w in zip(self.means, self.covs, self.log_w)]), axis = 0)
        return p

    @from_probit
    def sample_from_dpgmm(self, n_samps):
        """
        Draw samples from mixture
        
        Arguments:
            :int n_samps: number of samples to draw
        
        Returns:
            :np.ndarray: samples
        """
        idx = np.random.choice(np.arange(self.n_cl), p = self.w, size = int(n_samps))
        ctr = Counter(idx)
        if self.dim > 1:
            samples = np.empty(shape = (1,self.dim))
            for i, n in zip(ctr.keys(), ctr.values()):
                samples = np.concatenate((samples, np.atleast_2d(mn(self.means[i], self.covs[i]).rvs(size = n))))
        else:
            samples = np.array([np.zeros(1)])
            for i, n in zip(ctr.keys(), ctr.values()):
                samples = np.concatenate((samples, np.atleast_2d(mn(self.means[i], self.covs[i]).rvs(size = n)).T))
        return np.array(samples[1:])

    def _sample_from_dpgmm_probit(self, n_samps):
        """
        Draw samples from mixture in probit space
        
        Arguments:
            :int n_samps: number of samples to draw
        
        Returns:
            :np.ndarray: samples in probit space
        """
        idx = np.random.choice(np.arange(self.n_cl), p = self.w, size = n_samps)
        ctr = Counter(idx)
        if self.dim > 1:
            samples = np.empty(shape = (1,self.dim))
            for i, n in zip(ctr.keys(), ctr.values()):
                samples = np.concatenate((samples, np.atleast_2d(mn(self.means[i], self.covs[i]).rvs(size = n))))
        else:
            samples = np.array([np.zeros(1)])
            for i, n in zip(ctr.keys(), ctr.values()):
                samples = np.concatenate((samples, np.atleast_2d(mn(self.means[i], self.covs[i]).rvs(size = n)).T))
        return np.array(samples[1:])
        
#-------------------#
# Inference classes #
#-------------------#

class DPGMM:
    """
    Class to infer a distribution given a set of samples.
    
    Arguments:
        :iterable bounds:        boundaries of the rectangle over which the distribution is defined. It should be in the format [[xmin, xmax],[ymin, ymax],...]
        :iterable prior_pars:    NIG/NIW prior parameters (k, L, nu, mu)
        :double alpha0:          initial guess for concentration parameter
        :str or Path out_folder: folder for outputs
        :int n_draws_norm:       number of MC draws to estimate normalisation constant while instancing mixture class
    
    Returns:
        :DPGMM: instance of DPGMM class
    """
    def __init__(self, bounds,
                       prior_pars = None,
                       alpha0     = 1.,
                       out_folder = '.',
                       n_draws_norm = 1000,
                       ):
        self.bounds   = np.array(bounds)
        self.dim      = len(self.bounds)
        if prior_pars is not None:
            self.prior = prior(*prior_pars)
        else:
            self.prior = prior(1e-1, np.identity(self.dim)*0.2**2, self.dim, np.zeros(self.dim))
        self.alpha      = alpha0
        self.alpha_0    = alpha0
        self.mixture    = []
        self.N_list     = []
        self.n_cl       = 0
        self.n_pts      = 0
        self.n_draws_norm = n_draws_norm
    
    def initialise(self, prior_pars = None):
        """
        Initialise the mixture to initial conditions.
        
        Arguments:
            :iterable prior_pars: NIG/NIW prior parameters (k, L, nu, mu). If None, old parameters are kept
        """
        self.alpha    = self.alpha_0
        self.mixture  = []
        self.N_list   = []
        self.n_cl     = 0
        self.n_pts    = 0
        if prior_pars is not None:
            self.prior = prior(*prior_pars)
        
    def _add_datapoint_to_component(self, x, ss):
        """
        Update component parameters after assigning a sample to a component
        
        Arguments:
            :np.ndarray x: sample
            :component ss: component to update
        
        Returns:
            :component: updated component
        """
        new_mean, new_cov, new_N, new_mu, new_sigma = compute_component_suffstats(x, ss.mean, ss.cov, ss.N, self.prior.mu, self.prior.k, self.prior.nu, self.prior.L)
        ss.mean  = new_mean
        ss.cov   = new_cov
        ss.N     = new_N
        ss.mu    = new_mu
        ss.sigma = new_sigma
        return ss
    
    def _log_predictive_likelihood(self, x, ss):
        """
        Compute log likelihood of drawing sample x from component ss given the samples that are already assigned to that component.
        
        Arguments:
            :np.ndarray x: sample
            :component ss: component to update
        
        Returns:
            :double: log Likelihood
        """
        if ss == "new":
            ss = component(np.zeros(self.dim), prior = self.prior)
            ss.N = 0.
        t_df, t_shape, mu_n = compute_t_pars(self.prior.k, self.prior.mu, self.prior.nu, self.prior.L, ss.mean, ss.cov, ss.N, self.dim)
        return student_t(df = t_df, t = x, mu = mu_n, sigma = t_shape, dim = self.dim)

    def _cluster_assignment_distribution(self, x):
        """
        Compute the marginal distribution of cluster assignment for each cluster.
        
        Arguments:
            :np.ndarray x: sample
        
        Returns:
            :dict: p_i for each component
        """
        scores = {}
        for i in list(np.arange(self.n_cl)) + ["new"]:
            if i == "new":
                ss = "new"
            else:
                ss = self.mixture[i]
            scores[i] = self._log_predictive_likelihood(x, ss)
            if ss == "new":
                scores[i] += np.log(self.alpha)
            else:
                scores[i] += np.log(ss.N)
        scores = {cid: np.exp(score) for cid, score in scores.items()}
        normalization = 1/sum(scores.values())
        scores = {cid: score*normalization for cid, score in scores.items()}
        return scores

    def _assign_to_cluster(self, x):
        """
        Assign the new sample x to an existing cluster or to a new cluster according to the marginal distribution of cluster assignment.
        
        Arguments:
            :np.ndarray x: sample
        """
        scores = self._cluster_assignment_distribution(x).items()
        labels, scores = zip(*scores)
        cid = np.random.choice(labels, p=scores)
        if cid == "new":
            self.mixture.append(component(x, prior = self.prior))
            self.N_list.append(1.)
            self.n_cl += 1
        else:
            self.mixture[int(cid)] = self._add_datapoint_to_component(x, self.mixture[int(cid)])
            self.N_list[int(cid)] += 1
        # Update weights
        self.w = np.array(self.N_list)
        self.w = self.w/self.w.sum()
        self.log_w = np.log(self.w)
        return
    
    def density_from_samples(self, samples):
        """
        Reconstruct the probability density from a set of samples.
        
        Arguments:
            :iterable samples: samples set
        """
        for s in samples:
            self.add_new_point(np.atleast_2d(s))
    
    @probit
    def add_new_point(self, x):
        """
        Update the probability density reconstruction adding a new sample
        
        Arguments:
            :np.ndarray x: sample
        """
        self.n_pts += 1
        self.assign_to_cluster(np.atleast_2d(x))
        self.alpha = update_alpha(self.alpha, self.n_pts, self.n_cl)
    
    def sample_from_dpgmm(self, n_samps):
        """
        Draw samples from mixture
        
        Arguments:
            :int n_samps: number of samples to draw
        
        Returns:
            :np.ndarray: samples
        """
        idx = np.random.choice(np.arange(self.n_cl), p = self.w, size = n_samps)
        ctr = Counter(idx)
        if self.dim > 1:
            samples = np.empty(shape = (1,self.dim))
            for i, n in zip(ctr.keys(), ctr.values()):
                samples = np.concatenate((samples, np.atleast_2d(mn(self.mixture[i].mu, self.mixture[i].sigma).rvs(size = n))))
        else:
            samples = np.array([np.zeros(1)])
            for i, n in zip(ctr.keys(), ctr.values()):
                samples = np.concatenate((samples, np.atleast_2d(mn(self.mixture[i].mu, self.mixture[i].sigma).rvs(size = n)).T))
        return samples[1:]

    def _sample_from_dpgmm_probit(self, n_samps):
        """
        Draw samples from mixture in probit space
        
        Arguments:
            :int n_samps: number of samples to draw
        
        Returns:
            :np.ndarray: samples in probit space
        """
        idx = np.random.choice(np.arange(self.n_cl), p = self.w, size = n_samps)
        ctr = Counter(idx)
        if self.dim > 1:
            samples = np.empty(shape = (1,self.dim))
            for i, n in zip(ctr.keys(), ctr.values()):
                samples = np.concatenate((samples, np.atleast_2d(mn(self.mixture[i].mu, self.mixture[i].sigma).rvs(size = n))))
        else:
            samples = np.array([np.zeros(1)])
            for i, n in zip(ctr.keys(), ctr.values()):
                samples = np.concatenate((samples, np.atleast_2d(mn(self.mixture[i].mu, self.mixture[i].sigma).rvs(size = n)).T))
        return samples[1:]

    def _evaluate_mixture_in_probit(self, x):
        """
        Evaluate mixture at point(s) x in probit space
        
        Arguments:
            :np.ndarray x: point(s) to evaluate the mixture at (in probit space)
        
        Returns:
            :np.ndarray: mixture.pdf(x)
        """
        p = np.sum(np.array([w*mn(comp.mu, comp.sigma).pdf(x) for comp, w in zip(self.mixture, self.w)]), axis = 0)
        return p

    @probit
    def _evaluate_mixture_no_jacobian(self, x):
        """
        Evaluate mixture at point(s) x without jacobian
        
        Arguments:
            :np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            :np.ndarray: mixture.pdf(x)
        """
        p = np.sum(np.array([w*mn(comp.mu, comp.sigma).pdf(x) for comp, w in zip(self.mixture, self.w)]), axis = 0)
        return p
    
    @probit
    def evaluate_mixture(self, x):
        """
        Evaluate mixture at point(s) x
        
        Arguments:
            :np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            :np.ndarray: mixture.pdf(x)
        """
        p = np.sum(np.array([w*mn(comp.mu, comp.sigma).pdf(x) for comp, w in zip(self.mixture, self.w)]), axis = 0)
        return p * np.exp(-probit_logJ(x, self.bounds))

    def _evaluate_log_mixture_in_probit(self, x):
        """
        Evaluate log mixture at point(s) x in probit space
        
        Arguments:
            :np.ndarray x: point(s) to evaluate the mixture at (in probit space)
        
        Returns:
            :np.ndarray: mixture.logpdf(x)
        """
        p = logsumexp(np.array([w + mn(comp.mu, comp.sigma).logpdf(x) for comp, w in zip(self.mixture, self.log_w)]), axis = 0)
        return p

    @probit
    def _evaluate_log_mixture_no_jacobian(self, x):
        """
        Evaluate log mixture at point(s) x without jacobian
        
        Arguments:
            :np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            :np.ndarray: mixture.logpdf(x)
        """
        p = logsumexp(np.array([w + mn(comp.mu, comp.sigma).logpdf(x) for comp, w in zip(self.mixture, self.log_w)]), axis = 0)
        return p
        
    @probit
    def evaluate_log_mixture(self, x):
        """
        Evaluate mixture at point(s) x
        
        Arguments:
            :np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            :np.ndarray: mixture.pdf(x)
        """
        p = logsumexp(np.array([w + mn(comp.mu, comp.sigma).logpdf(x) for comp, w in zip(self.mixture, self.log_w)]), axis = 0)
        return p - probit_logJ(x, self.bounds)

    def save_density(self):
        """
        Build and save density
        """
        mixture = self.build_mixture()
        with open(Path(self.out_folder, 'mixture.pkl'), 'wb') as dill_file:
            dill.dump(mixture, dill_file)
        
    def build_mixture(self):
        """
        Instances a mixture class representing the inferred distribution
        
        Returns:
            :mixture: the inferred distribution
        """
        return mixture(np.array([comp.mu for comp in self.mixture]), np.array([comp.sigma for comp in self.mixture]), np.array(self.w), self.bounds, self.dim, self.n_cl, self.n_pts, n_draws = self.n_draws_norm)


class HDPGMM(DPGMM):
    """
    Class to infer a distribution given a set of observations (each being a set of samples).
    Child of DPGMM class
    
    Arguments:
        :iterable bounds:        boundaries of the rectangle over which the distribution is defined. It should be in the format [[xmin, xmax],[ymin, ymax],...]
        :iterable prior_pars:    NIG/NIW prior parameters (k, L, nu, mu)
        :double alpha0:          initial guess for concentration parameter
        :str or Path out_folder: folder for outputs
        :int n_draws_norm:       number of MC draws to estimate normalisation constant while instancing mixture class
    
    Returns:
        :HDPGMM: instance of HDPGMM class
    """
    def __init__(self, bounds,
                       alpha0     = 1.,
                       out_folder = '.',
                       prior_pars = None,
                       MC_draws   = 1e3,
                       n_draws_norm = 1000
                       ):
        dim = len(bounds)
        if prior_pars == None:
            prior_pars = (1e-1, np.identity(dim)*0.2**2, dim, np.zeros(dim))
        super().__init__(bounds = bounds, prior_pars = prior_pars, alpha0 = alpha0, out_folder = out_folder, n_draws_norm = n_draws_norm)
        self.MC_draws = int(MC_draws)
    
    def _add_new_point(self, ev):
        """
        Update the probability density reconstruction adding a new sample
        
        Arguments:
            :iterable x: set of single-event draws from a DPGMM inference
        """
        self.n_pts += 1
        x = np.random.choice(ev)
        self.assign_to_cluster(x)
        self.alpha = update_alpha(self.alpha, self.n_pts, self.n_cl)

    def _cluster_assignment_distribution(self, x):
        """
        Compute the marginal distribution of cluster assignment for each cluster.
        
        Arguments:
            :np.ndarray x: sample
        
        Returns:
            :dict: p_i for each component
        """
        scores = {}
        logL_N = {}
        for i in list(np.arange(self.n_cl)) + ["new"]:
            if i == "new":
                ss = "new"
            else:
                ss = self.mixture[i]
            scores[i], logL_N[i] = self._log_predictive_likelihood(x, ss)
            if ss == "new":
                scores[i] += np.log(self.alpha)
            else:
                scores[i] += np.log(ss.N)
        scores = {cid: np.exp(score) for cid, score in scores.items()}
        normalization = 1/sum(scores.values())
        scores = {cid: score*normalization for cid, score in scores.items()}
        return scores, logL_N

    def _assign_to_cluster(self, x):
        """
        Assign the new sample x to an existing cluster or to a new cluster according to the marginal distribution of cluster assignment.
        
        Arguments:
            :np.ndarray x: sample
        """
        scores, logL_N = self._cluster_assignment_distribution(x)
        scores = scores.items()
        labels, scores = zip(*scores)
        cid = np.random.choice(labels, p=scores)
        if cid == "new":
            self.mixture.append(component_h(x, self.dim, self.prior, logL_N[cid]))
            self.N_list.append(1.)
            self.n_cl += 1
        else:
            self.mixture[int(cid)] = self._add_datapoint_to_component(x, self.mixture[int(cid)], logL_N[int(cid)])
            self.N_list[int(cid)] += 1
            
        # Update weights
        self.w = np.array(self.N_list)
        self.w = self.w/self.w.sum()
        self.log_w = np.log(self.w)
        return

    def _log_predictive_likelihood(self, x, ss):
        """
        Compute log likelihood of drawing sample x from component ss given the samples that are already assigned to that component.
        
        Arguments:
            :np.ndarray x: sample
            :component ss: component to update
        
        Returns:
            :double: log Likelihood
        """
        if ss == "new":
            ss     = component(np.zeros(self.dim), prior = self.prior)
            events = []
            ss.N   = 0.
            logL_D = 0.
        else:
            events = ss.events
            logL_D = ss.logL_D
            
        events.append(x)
        
        if self.dim == 1:
            logL_N = MC_predictive_1d(events, n_samps = self.MC_draws, a = 2, b = self.prior.L[0,0])
        else:
            logL_N = MC_predictive(events, self.dim, n_samps = self.MC_draws, a = self.prior.nu, b = self.prior.L)
        return logL_N - logL_D, logL_N

    def _add_datapoint_to_component(self, x, ss, logL_D):
        """
        Update component parameters after assigning a sample to a component
        
        Arguments:
            :np.ndarray x: sample
            :component ss: component to update
            :double logL_D: log Likelihood denominator
        
        Returns:
            :component: updated component
        """
        ss.events.append(x)
        ss.means.append(x.means)
        ss.covs.append(x.covs)
        ss.log_w.append(x.log_w)
        ss.logL_D = logL_D
        
        if self.dim == 1:
            sample = sample_point_1d(ss.means, ss.covs, ss.log_w, a = self.prior.nu+1, b = self.prior.L[0,0])
        else:
            sample = sample_point(ss.means, ss.covs, ss.log_w, self.dim, a = self.prior.nu, b = self.prior.L)
        ss.mu, ss.sigma = build_mean_cov(sample, self.dim)
        ss.N += 1
        return ss

    def density_from_samples(self, events):
        """
        Reconstruct the probability density from a set of samples.
        
        Arguments:
            :iterable samples: set of single-event draws from DPGMM
        """
        for ev in events:
            self.add_new_point(ev)

    # Overwrites parent function to account for hierarchical issues
    def build_mixture(self):
        """
        Instances a mixture class representing the inferred distribution
        
        Returns:
            :mixture: the inferred distribution
        """
        return mixture(np.array([comp.mu for comp in self.mixture]), np.array([comp.sigma for comp in self.mixture]), np.array(self.w), self.bounds, self.dim, self.n_cl, self.n_pts, n_draws = self.n_draws_norm, hier_flag = True)
