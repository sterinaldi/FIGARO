import numpy as np
import sys
import dill

from collections import Counter
from pathlib import Path

from scipy.special import gammaln, logsumexp
from scipy.stats import multivariate_normal as mn
from scipy.stats import invgamma, invwishart

from figaro.decorators import *
from figaro.transform import *
from figaro.montecarlo import evaluate_mixture_MC_draws, evaluate_mixture_MC_draws_1d
from figaro.exceptions import except_hook

from numba import jit, njit, prange
from numba.extending import get_cython_function_address
import ctypes

#-----------#
# Utilities #
#-----------#

sys.excepthook = except_hook

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
    L_n  = L + S*N + k*N*((mean - mu).T@(mean - mu))/k_n
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
    new_sigma = (p_L + new_cov*new_N + p_k*new_N*((new_mean - p_mu).T@(new_mean - p_mu))/(p_k + new_N))/(p_nu + new_N - x.shape[-1] - 1)
    
    return new_mean, new_cov, new_N, new_mu, new_sigma

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
        self.k  = k
        self.L  = L*(nu-mu.shape[-1]-1)
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
        self.sigma = np.identity(x.shape[-1]).astype(np.float64)*prior.L/(prior.nu - x.shape[-1] - 1)

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
    def __init__(self, x, dim, prior, logL_D, mu_MC, sigma_MC):
        self.dim    = dim
        self.N      = 1
        self.events = [x]
        self.means  = [x.means]
        self.covs   = [x.covs]
        self.log_w  = [x.log_w]
        self.logL_D = logL_D
        
        self.mu    = np.average(mu_MC, weights = np.exp(logL_D), axis = 0)
        self.sigma = np.average(sigma_MC, weights = np.exp(logL_D), axis = 0)
        if dim == 1:
            self.mu = np.atleast_2d(self.mu).T
            self.sigma = np.atleast_2d(self.sigma).T
            
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
        :bool hier_flag:    flag for hierarchical mixture (needed to fix an issue with means)
    
    Returns:
        :mixture: instance of mixture class
    """
    def __init__(self, means, covs, w, bounds, dim, n_cl, n_pts, hier_flag = False):
        self.means  = means
        self.covs   = covs
        self.w      = w
        self.log_w  = np.log(w)
        self.bounds = bounds
        self.dim    = dim
        self.n_cl   = n_cl
        self.n_pts  = n_pts
        
    @probit
    def evaluate_mixture(self, x):
        """
        Evaluate mixture at point(s) x
        
        Arguments:
            :np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            :np.ndarray: mixture.pdf(x)
        """
        p = np.sum(np.array([w*mn(mean, cov).pdf(x) for mean, cov, w in zip(self.means, self.covs, self.w)]), axis = 0)
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
        p = logsumexp(np.array([w + mn(mean, cov).logpdf(x) for mean, cov, w in zip(self.means, self.covs, self.log_w)]), axis = 0)
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
    
    Returns:
        :DPGMM: instance of DPGMM class
    """
    def __init__(self, bounds,
                       prior_pars = None,
                       alpha0     = 1.,
                       out_folder = '.',
                       ):
        self.bounds   = np.array(bounds)
        self.dim      = len(self.bounds)
        if prior_pars is not None:
            self.prior = prior(*prior_pars)
        else:
            self.prior = prior(1e-2, np.identity(self.dim)*0.2**2, self.dim+2, np.zeros(self.dim))
        self.alpha      = alpha0
        self.alpha_0    = alpha0
        self.mixture    = []
        self.N_list     = []
        self.n_cl       = 0
        self.n_pts      = 0
    
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
        self._assign_to_cluster(np.atleast_2d(x))
        self.alpha = update_alpha(self.alpha, self.n_pts, self.n_cl)
    
    @from_probit
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
        return mixture(np.array([comp.mu for comp in self.mixture]), np.array([comp.sigma for comp in self.mixture]), np.array(self.w), self.bounds, self.dim, self.n_cl, self.n_pts)


class HDPGMM(DPGMM):
    """
    Class to infer a distribution given a set of observations (each being a set of samples).
    Child of DPGMM class
    
    Arguments:
        :iterable bounds:        boundaries of the rectangle over which the distribution is defined. It should be in the format [[xmin, xmax],[ymin, ymax],...]
        :iterable prior_pars:    NIG/NIW prior parameters (k, L, nu, mu)
        :double alpha0:          initial guess for concentration parameter
        :str or Path out_folder: folder for outputs
    
    Returns:
        :HDPGMM: instance of HDPGMM class
    """
    def __init__(self, bounds,
                       alpha0     = 1.,
                       out_folder = '.',
                       prior_pars = None,
                       MC_draws   = 1e3,
                       ):
        dim = len(bounds)
        if prior_pars == None:
            prior_pars = (1e-2, np.identity(dim)*0.2**2, dim+2, np.zeros(dim))
        super().__init__(bounds = bounds, prior_pars = prior_pars, alpha0 = alpha0, out_folder = out_folder)
        self.MC_draws = int(MC_draws)
        if self.dim == 1:
            self.sigma_MC = invgamma(self.prior.nu/2, scale = self.prior.nu*self.prior.L[0,0]/2.).rvs(size = self.MC_draws)
            self.mu_MC    = np.array([np.random.normal(loc = self.prior.mu[0], scale = s) for s in np.sqrt(self.sigma_MC/self.prior.k)])
        else:
            df = np.max([self.prior.nu, dim + 2])
            self.sigma_MC = invwishart(df = df, scale = self.prior.L*(df-dim-1)).rvs(size = self.MC_draws)
            self.mu_MC    = np.array([mn(self.prior.mu, s/self.prior.k).rvs() for s in self.sigma_MC])
        
    def initialise(self, prior_pars = None):
        super().initialise(prior_pars = prior_pars)
        if self.dim == 1:
            self.sigma_MC = invgamma(self.prior.nu/2, scale = self.prior.nu*self.prior.L[0,0]/2.).rvs(size = self.MC_draws)
            self.mu_MC    = np.array([np.random.normal(loc = self.prior.mu[0], scale = s) for s in np.sqrt(self.sigma_MC/self.prior.k)])
        else:
            df = np.max([self.prior.nu, dim + 2])
            self.sigma_MC = invwishart(df = df, scale = self.prior.L*(df-dim-1)).rvs(size = self.MC_draws)
            self.mu_MC    = np.array([mn(self.prior.mu, s/self.prior.k).rvs() for s in self.sigma_MC])
    
    def add_new_point(self, ev):
        """
        Update the probability density reconstruction adding a new sample
        
        Arguments:
            :iterable x: set of single-event draws from a DPGMM inference
        """
        self.n_pts += 1
        x = np.random.choice(ev)
        self._assign_to_cluster(x)
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
        
        if self.dim == 1:
            logL_x = evaluate_mixture_MC_draws_1d(self.mu_MC, self.sigma_MC, x.means, x.covs, x.w)
        else:
            logL_x = evaluate_mixture_MC_draws(self.mu_MC, self.sigma_MC, x.means, x.covs, x.w)

        for i in list(np.arange(self.n_cl)) + ["new"]:
            if i == "new":
                ss = "new"
                logL_D = np.zeros(self.MC_draws)
            else:
                ss = self.mixture[i]
                logL_D = ss.logL_D
            scores[i] = logsumexp(logL_D + logL_x) - logsumexp(logL_D)
            logL_N[i] = logL_D + logL_x
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
            self.mixture.append(component_h(x, self.dim, self.prior, logL_N[cid], self.mu_MC, self.sigma_MC))
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
        
        log_norm = logsumexp(logL_D)

        ss.mu    = np.average(self.mu_MC, weights = np.exp(logL_D - log_norm), axis = 0)
        ss.sigma = np.average(self.sigma_MC, weights = np.exp(logL_D - log_norm), axis = 0)
        if self.dim == 1:
            ss.mu = np.atleast_2d(ss.mu).T
            ss.sigma = np.atleast_2d(ss.sigma).T
        
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
