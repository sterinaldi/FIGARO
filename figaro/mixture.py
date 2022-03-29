import numpy as np
import sys
import dill

from collections import Counter
from pathlib import Path

from scipy.special import gammaln, logsumexp
from scipy.stats import multivariate_normal as mn
from scipy.stats import invwishart

import cpnest.model

from figaro.decorators import *
from figaro.transform import *
from figaro.metropolis import sample_point, Integrator, MC_predictive_1d, MC_predictive
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

@jit
def log_add(x, y):
     if x >= y:
        return x+np.log1p(np.exp(y-x))
     else:
        return y+np.log1p(np.exp(x-y))

@njit
def numba_gammaln(x):
    return gammaln_float64(x)

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
def student_t(df, t, mu, sigma, dim):
    """
    http://gregorygundersen.com/blog/2020/01/20/multivariate-t/
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
    # Update hyperparameters
    k_n, mu_n, nu_n, L_n = compute_hyperpars(k, mu, nu, L, mean, S, N)
    # Update t-parameters
    t_df    = nu_n - dim + 1
    t_shape = L_n*(k_n+1)/(k_n*t_df)
    return t_df, t_shape, mu_n

@jit
def compute_hyperpars(k, mu, nu, L, mean, S, N):
    k_n  = k + N
    mu_n = (mu*k + N*mean)/k_n
    nu_n = nu + N
    L_n  = L*k + S*N + k*N*((mean - mu).T@(mean - mu))/k_n
    return k_n, mu_n, nu_n, L_n

@jit
def compute_component_suffstats(x, mean, cov, N, mu, sigma, p_mu, p_k, p_nu, p_L):

    new_mean  = (mean*N+x)/(N+1)
    new_cov   = (N*(cov + mean.T@mean) + x.T@x)/(N+1) - new_mean.T@new_mean
    new_N     = N+1
    new_mu    = ((p_mu*p_k + new_N*new_mean)/(p_k + new_N))[0]
    new_sigma = (p_L*p_k + new_cov*new_N + p_k*new_N*((new_mean - p_mu).T@(new_mean - p_mu))/(p_k + new_N))/(p_nu + new_N)
    
    return new_mean, new_cov, new_N, new_mu, new_sigma


def build_mean_cov(x, dim):
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

class component:
    def __init__(self, x, prior):
        self.N     = 1
        self.mean  = x
        self.cov   = np.identity(x.shape[-1])*0.
        self.mu    = np.atleast_2d((prior.mu*prior.k + self.N*self.mean)/(prior.k + self.N)).astype(np.float64)[0]
        self.sigma = np.identity(x.shape[-1]).astype(np.float64)*prior.L

class component_h:
    def __init__(self, x, dim, prior, MC_draws, logL_D):
        self.dim    = dim
        self.N      = 1
        self.events = [x]
        self.means  = [x.means]
        self.covs   = [x.covs]
        self.log_w  = [x.log_w]
        self.logL_D = logL_D
        
        if self.dim == 1:
            sample = sample_point(self.means, self.covs, self.log_w, a = 2, b = prior.L[0,0])
        else:
            integrator = cpnest.CPNest(Integrator(self.means, self.covs, self.dim, prior.nu, prior.L),
                                            verbose = 0,
                                            nlive   = int(2*self.dim +self.dim*(self.dim-1)/2.),
                                            maxmcmc = 100,
                                            nensemble = 1,
                                            output = Path('.'),
                                            )
            integrator.run()
            sample = np.array(integrator.posterior_samples[-1].tolist())[:-2]
        self.mu, self.sigma = build_mean_cov(sample, self.dim)
    
class mixture:
    def __init__(self, means, covs, w, bounds, dim, n_cl, n_pts, n_draws = 1000):
        if dim > 1:
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
    
    def _compute_norm_const(self, n_draws):
        p_ss     = self.sample_from_dpgmm(n_draws)
        min_vals = np.atleast_1d(p_ss.min(axis = 0))
        max_vals = np.atleast_1d(p_ss.max(axis = 0))
        volume   = np.prod(np.diff(np.array([min_vals, max_vals]).T))
        ss       = np.random.uniform(min_vals, max_vals, size = (n_draws, self.dim))
        return self.evaluate_mixture(np.atleast_2d(ss)).sum()*volume/n_draws
        
    @probit
    def evaluate_mixture(self, x):
        p = np.sum(np.array([w*mn(mean, cov).pdf(x)/self.norm for mean, cov, w in zip(self.means, self.covs, self.w)]), axis = 0)
        return p * np.exp(-probit_logJ(x, self.bounds))

    @probit
    def evaluate_log_mixture(self, x):
        p = logsumexp(np.array([w + mn(mean, cov).logpdf(x) - self.log_norm for mean, cov, w in zip(self.means, self.covs, self.log_w)]), axis = 0)
        return p - probit_logJ(x, self.bounds)
        
    def _evaluate_mixture_in_probit(self, x):
        p = np.sum(np.array([w*mn(mean, cov).pdf(x) for mean, cov, w in zip(self.means, self.covs, self.w)]), axis = 0)
        return p

    def _evaluate_log_mixture_in_probit(self, x):
        p = logsumexp(np.array([w + mn(mean, cov).logpdf(x) for mean, cov, w in zip(self.means, self.covs, self.log_w)]), axis = 0)
        return p

    @from_probit
    def sample_from_dpgmm(self, n_samps):
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

class prior:
    def __init__(self, k, L, nu, mu):
        self.k = k
        self.L = L
        self.mu = mu
        self.nu = nu
        
#-------------------#
# Inference classes #
#-------------------#

class DPGMM:
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
            self.prior = prior(1e-3, np.identity(self.dim)*0.2**2, self.dim, np.zeros(self.dim))
        self.alpha      = alpha0
        self.alpha_0    = alpha0
        self.mixture    = []
        self.N_list     = []
        self.n_cl       = 0
        self.n_pts      = 0
        self.n_draws_norm = n_draws_norm
    
    def initialise(self, prior_pars = None):
        self.alpha = self.alpha_0
        self.mixture  = []
        self.N_list   = []
        self.n_cl     = 0
        self.n_pts    = 0
        if prior_pars is not None:
            self.prior = prior(*prior_pars)
        
    def add_datapoint_to_component(self, x, ss):
        new_mean, new_cov, new_N, new_mu, new_sigma = compute_component_suffstats(x, ss.mean, ss.cov, ss.N, ss.mu, ss.sigma, self.prior.mu, self.prior.k, self.prior.nu, self.prior.L)
        ss.mean  = new_mean
        ss.cov   = new_cov
        ss.N     = new_N
        ss.mu    = new_mu
        ss.sigma = new_sigma
        return ss
    
    def log_predictive_likelihood(self, x, ss):
        if ss == "new":
            ss = component(np.zeros(self.dim), prior = self.prior)
            ss.N = 0.
        t_df, t_shape, mu_n = compute_t_pars(self.prior.k, self.prior.mu, self.prior.nu, self.prior.L, ss.mean, ss.cov, ss.N, self.dim)
        return student_t(df = t_df, t = x, mu = mu_n, sigma = t_shape, dim = self.dim)

    def cluster_assignment_distribution(self, x):
        scores = {}
        for i in list(np.arange(self.n_cl)) + ["new"]:
            if i == "new":
                ss = "new"
            else:
                ss = self.mixture[i]
            scores[i] = self.log_predictive_likelihood(x, ss)
            if ss == "new":
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
            self.N_list.append(1.)
            self.n_cl += 1
        else:
            self.mixture[int(cid)] = self.add_datapoint_to_component(x, self.mixture[int(cid)])
            self.N_list[int(cid)] += 1
        # Update weights
        self.w = np.array(self.N_list)
        self.w = self.w/self.w.sum()
        self.log_w = np.log(self.w)
        return
    
    def density_from_samples(self, samples):
        for s in samples:
            self.add_new_point(np.atleast_2d(s))
    
    @probit
    def add_new_point(self, x):
        self.n_pts += 1
        self.assign_to_cluster(np.atleast_2d(x))
        self.alpha = update_alpha(self.alpha, self.n_pts, self.n_cl)
    
    @from_probit
    def sample_from_dpgmm(self, n_samps):
        idx = np.random.choice(np.arange(self.n_cl), p = self.w, size = n_samps)
        ctr = Counter(idx)
        samples = np.empty(shape = (1,self.dim))
        for i, n in zip(ctr.keys(), ctr.values()):
            samples = np.concatenate((samples, np.atleast_2d(mn(self.mixture[i].mu, self.mixture[i].sigma).rvs(size = n))))
        return samples[1:]
    
    def _sample_from_dpgmm_probit(self, n_samps):
        idx = np.random.choice(np.arange(self.n_cl), p = self.w, size = n_samps)
        ctr = Counter(idx)
        samples = np.empty(shape = (1,self.dim))
        for i, n in zip(ctr.keys(), ctr.values()):
            samples = np.concatenate((samples, np.atleast_2d(mn(self.mixture[i].mu, self.mixture[i].sigma).rvs(size = n))))
        return samples[1:]

    def _evaluate_mixture_in_probit(self, x):
        p = np.sum(np.array([w*mn(comp.mu, comp.sigma).pdf(x) for comp, w in zip(self.mixture, self.w)]), axis = 0)
        return p

    @probit
    def evaluate_mixture(self, x):
        p = np.sum(np.array([w*mn(comp.mu, comp.sigma).pdf(x) for comp, w in zip(self.mixture, self.w)]), axis = 0)
        return p
    
    @probit
    def evaluate_mixture_with_jacobian(self, x):
        p = np.sum(np.array([w*mn(comp.mu, comp.sigma).pdf(x) for comp, w in zip(self.mixture, self.w)]), axis = 0)
        return p * np.exp(-probit_logJ(x, self.bounds))

    def _evaluate_log_mixture_in_probit(self, x):
        p = logsumexp(np.array([w + mn(comp.mu, comp.sigma).logpdf(x) for comp, w in zip(self.mixture, self.log_w)]), axis = 0)
        return p

    @probit
    def evaluate_log_mixture(self, x):
        p = logsumexp(np.array([w + mn(comp.mu, comp.sigma).logpdf(x) for comp, w in zip(self.mixture, self.log_w)]), axis = 0)
        return p
        
    @probit
    def evaluate_log_mixture_with_jacobian(self, x):
        p = logsumexp(np.array([w + mn(comp.mu, comp.sigma).logpdf(x) for comp, w in zip(self.mixture, self.log_w)]), axis = 0)
        return p - probit_logJ(x, self.bounds)

    def save_density(self):
        mixture = self.build_mixture()
        with open(Path(self.out_folder, 'mixture.pkl'), 'wb') as dill_file:
            dill.dump(mixture, dill_file)
        
    def build_mixture(self):
        return mixture(np.array([comp.mu for comp in self.mixture]), np.array([comp.sigma for comp in self.mixture]), np.array(self.w), self.bounds, self.dim, self.n_cl, self.n_pts, n_draws = self.n_draws_norm)


class HDPGMM(DPGMM):
    def __init__(self, bounds,
                       alpha0     = 1.,
                       out_folder = '.',
                       prior_pars = None,
                       MC_draws   = 1e3,
                       n_draws_norm = 1000
                       ):
        dim = len(bounds)
        if prior_pars == None:
            prior_pars = (1e-3, np.identity(dim)*0.2**2, dim, np.zeros(dim))
        super().__init__(bounds = bounds, prior_pars = prior_pars, alpha0 = alpha0, out_folder = out_folder, n_draws_norm = n_draws_norm)
        self.MC_draws = int(MC_draws)
    
    def add_new_point(self, ev):
        self.n_pts += 1
        x = np.random.choice(ev)
        self.assign_to_cluster(x)
        self.alpha = update_alpha(self.alpha, self.n_pts, self.n_cl)

    def cluster_assignment_distribution(self, x):
        scores = {}
        logL_N = {}
        for i in list(np.arange(self.n_cl)) + ["new"]:
            if i == "new":
                ss = "new"
            else:
                ss = self.mixture[i]
            scores[i], logL_N[i] = self.log_predictive_likelihood(x, ss)
            if ss == "new":
                scores[i] += np.log(self.alpha)
            else:
                scores[i] += np.log(ss.N)
        scores = {cid: np.exp(score) for cid, score in scores.items()}
        normalization = 1/sum(scores.values())
        scores = {cid: score*normalization for cid, score in scores.items()}
        return scores, logL_N

    def assign_to_cluster(self, x):
        scores, logL_N = self.cluster_assignment_distribution(x)
        scores = scores.items()
        labels, scores = zip(*scores)
        cid = np.random.choice(labels, p=scores)
        if cid == "new":
            self.mixture.append(component_h(x, self.dim, self.prior, self.MC_draws, logL_N[cid]))
            self.N_list.append(1.)
            self.n_cl += 1
        else:
            self.mixture[int(cid)] = self.add_datapoint_to_component(x, self.mixture[int(cid)], logL_N[int(cid)])
            self.N_list[int(cid)] += 1
            
        # Update weights
        self.w = np.array(self.N_list)
        self.w = self.w/self.w.sum()
        self.log_w = np.log(self.w)
        return

    def log_predictive_likelihood(self, x, ss):
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

    def add_datapoint_to_component(self, x, ss, logL_D):
        ss.events.append(x)
        ss.means.append(x.means)
        ss.covs.append(x.covs)
        ss.log_w.append(x.log_w)
        ss.logL_D = logL_D
        
        if self.dim == 1:
            sample = sample_point(ss.means, ss.covs, ss.log_w, a = 2, b = self.prior.L[0,0])
        else:
            integrator = cpnest.CPNest(Integrator(ss.means, ss.covs, self.dim, self.prior.nu, self.prior.L),
                                            verbose = 0,
                                            nlive   = int(2*self.dim +self.dim*(self.dim-1)/2.),
                                            maxmcmc = 100,
                                            nensemble = 1,
                                            output = Path('.'),
                                            )
            integrator.run()
            sample = np.array(integrator.posterior_samples[-1].tolist())[:-2]

        ss.mu, ss.sigma = build_mean_cov(sample, self.dim)
        ss.N += 1
        return ss

    def density_from_samples(self, events):
        for ev in events:
            self.add_new_point(ev)
