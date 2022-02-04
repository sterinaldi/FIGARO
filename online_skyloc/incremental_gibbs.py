import numpy as np
from collections import Counter

from scipy.special import gammaln, logsumexp
from scipy.stats import multivariate_normal as mn

from pathlib import Path
import dill

from online_skyloc.decorators import *
from online_skyloc.transform import *

from numba import jit, njit
from numba.extending import get_cython_function_address
import ctypes

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
functype = ctypes.CFUNCTYPE(_dble, _dble)
gammaln_float64 = functype(addr)

@njit
def numba_gammaln(x):
    return gammaln_float64(x)

class component:
    def __init__(self, x, prior):
        self.N     = 1
        self.mean  = x
        self.cov   = np.identity(x.shape[-1])*0.
        self.mu    = np.atleast_2d((prior.mu*prior.k + self.N*self.mean)/(prior.k + self.N))[0]
        self.sigma = np.identity(x.shape[-1])*prior.L
        self.w     = 0.

class prior:
    def __init__(self, k, L, nu, mu):
        self.k = k
        self.L = L
        self.mu = mu
        self.nu = nu

@jit
def student_t(df, t, mu, sigma, dim, s2max):
    """
    http://gregorygundersen.com/blog/2020/01/20/multivariate-t/
    """
    vals, vecs = np.linalg.eigh(sigma)
    vals       = np.minimum(vals, s2max)
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
def update_alpha(alpha, n, K, burnin = 100):
    a_old = alpha
    n_draws = burnin+np.random.randint(100)
    for i in range(n_draws):
        a_new = a_old + (np.random.random() - 0.5)
        if a_new > 0.:
            logP_old = numba_gammaln(a_old) - numba_gammaln(a_old + n) + K * np.log(a_old) - 1./a_old
            logP_new = numba_gammaln(a_new) - numba_gammaln(a_new + n) + K * np.log(a_new) - 1./a_new
            if logP_new - logP_old > np.log(np.random.random()):
                a_old = a_new
    return a_old

@jit
def compute_t_pars(k, mu, nu, L, mean, S, N, dim):
    # Update hyperparameters
    k_n  = k + N
    mu_n = (mu*k + N*mean)/k_n
    nu_n = nu + N
    L_n  = L*k + S*N + k*N*((mean - mu).T@(mean - mu))/k_n
    # Update t-parameters
    t_df    = nu_n - dim + 1
    t_shape = L_n*(k_n+1)/(k_n*t_df)
    return t_df, t_shape, mu_n

@jit
def compute_component_suffstats(x, mean, cov, N, mu, sigma, p_mu, p_k, p_nu, p_L):

    new_mean  = (mean*N+x)/(N+1)
    new_cov   = (N*(cov + mean.T@mean) + x.T@x)/(N+1) - new_mean.T@new_mean
    new_N     = N+1
    new_mu    = ((p_mu*p_k + new_N*new_mean)/(p_k + new_N))[0]
    new_sigma = (p_L*p_k + new_cov*new_N + p_k*new_N*((new_mean - p_mu).T@(new_mean - p_mu))/(p_k + new_N))/(p_nu + new_N)
    
    return new_mean, new_cov, new_N, new_mu, new_sigma

class mixture:
    def __init__(self, bounds,
                       prior_pars = None,
                       alpha0     = 1.,
                       sigma_max  = 0.05,
                       out_folder = '.'
                       ):
        self.bounds   = np.array(bounds)
        self.dim      = len(self.bounds)
        if prior_pars is not None:
            self.prior = prior(*prior_pars)
        else:
            self.prior = prior(1e-3, np.identity(self.dim)*0.5, self.dim, np.zeros(self.dim))
        self.alpha    = alpha0
        self.mixture  = []
        self.n_cl     = 0
        self.n_pts    = 0
        self.s2max    = (sigma_max)**2
        self.normalised = False
    
    def add_datapoint_to_component(self, x, ss):
        new_mean, new_cov, new_N, new_mu, new_sigma = compute_component_suffstats(x, ss.mean, ss.cov, ss.N, ss.mu, ss.sigma, self.prior.mu, self.prior.k, self.prior.nu, self.prior.L)
        ss.mean  = new_mean
        ss.cov   = new_cov
        ss.N     = new_N
        ss.mu    = new_mu
        ss.sigma = new_sigma
        return ss
    
    def log_predictive_likelihood(self, x, ss):
        dim = x.shape[-1]
        if ss == "new":
            ss = component(np.zeros(dim), prior = self.prior)
            ss.N = 0.
        t_df, t_shape, mu_n = compute_t_pars(self.prior.k, self.prior.mu, self.prior.nu, self.prior.L, ss.mean, ss.cov, ss.N, dim)
        return student_t(df = t_df, t = x, mu = mu_n, sigma = t_shape, dim = dim, s2max = self.s2max)

    def cluster_assignment_distribution(self, x):
        scores = {}
        for i in list(np.arange(self.n_cl)) + ["new"]:
            if i == "new":
                ss = "new"
            else:
                ss = self.mixture[i]
            scores[i] = self.log_predictive_likelihood(x, ss)
            if ss is "new":
                scores[i] += np.log(self.alpha)
            else:
                scores[i] += np.log(ss.N)
        scores = {cid: np.exp(score) for cid, score in scores.items()}
        normalization = 1/sum(scores.values())
        scores = {cid: score*normalization for cid, score in scores.items()}
        return scores

    def assign_to_cluster(self, x):
        scores = self.cluster_assignment_distribution(x).items()
        labels, scores = zip(*scores)
        cid = np.random.choice(labels, p=scores)
        if cid == "new":
            self.mixture.append(component(x, prior = self.prior))
            self.n_cl += 1
        else:
            self.mixture[int(cid)] = self.add_datapoint_to_component(x, self.mixture[int(cid)])
        return
    
    @probit
    def add_new_point(self, x):
        self.n_pts += 1
        self.assign_to_cluster(x)
        self.alpha = update_alpha(self.alpha, self.n_pts, self.n_cl)
    
    def normalise_mixture(self):
        for ss in self.mixture:
            ss.w = ss.N/self.n_pts
        self.w = np.array([ss.w for ss in self.mixture])
        self.log_w = np.log(self.w)
    
    @from_probit
    def sample_from_dpgmm(self, n_samps):
        if not self.normalised:
            self.normalise_mixture()
        idx = np.random.choice(np.arange(self.n_cl), p = self.w, size = n_samps)
        ctr = Counter(idx)
        samples = np.empty(shape = (1,self.dim))
        for i, n in zip(ctr.keys(), ctr.values()):
            samples = np.concatenate((samples, np.atleast_2d(mn(self.mixture[i].mu, self.mixture[i].sigma).rvs(size = n))))
        return samples[1:]

    def _evaluate_mixture_in_probit(self, x):
        self.normalise_mixture()
        p = np.sum([w*mn(comp.mu, comp.sigma).pdf(x) for comp, w in zip(self.mixture, self.w)], axis = 0)
        return p

    @probit
    def evaluate_mixture(self, x):
        self.normalise_mixture()
        p = np.sum([w*mn(comp.mu, comp.sigma).pdf(x) for comp, w in zip(self.mixture, self.w)], axis = 0)
        return p
    
    @probit
    def evaluate_mixture_with_jacobian(self, x):
        self.normalise_mixture()
        p = np.sum([w*mn(comp.mu, comp.sigma).pdf(x) for comp, w in zip(self.mixture, self.w)], axis = 0)
        return p * np.exp(-probit_logJ(x, self.bounds))

    def _evaluate_log_mixture_in_probit(self, x):
        self.normalise_mixture()
        p = logsumexp(np.array([w + mn(comp.mu, comp.sigma).logpdf(x) for comp, w in zip(self.mixture, self.log_w)]), axis = 0)
        return p

    @probit
    def evaluate_log_mixture(self, x):
        self.normalise_mixture()
        p = logsumexp(np.array([w + mn(comp.mu, comp.sigma).logpdf(x) for comp, w in zip(self.mixture, self.log_w)]), axis = 0)
        return p
        
    @probit
    def evaluate_log_mixture_with_jacobian(self, x):
        self.normalise_mixture()
        p = logsumexp(np.array([w + mn(comp.mu, comp.sigma).logpdf(x) for comp, w in zip(self.mixture, self.log_w)]), axis = 0)
        return p - probit_logJ(x, self.bounds)

    def save_density(self):
        with open(Path(self.out_folder, 'mixture.pkl'), 'wb') as dill_file:
            dill.dump(self, dill_file)
