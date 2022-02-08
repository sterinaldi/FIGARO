import numpy as np
from collections import Counter

from scipy.special import gammaln, logsumexp
from scipy.stats import multivariate_normal as mn
from scipy.stats import invwishart, multivariate_t
from scipy.integrate import dblquad

from pathlib import Path
import dill

import cpnest.model

from figaro.decorators import *
from figaro.transform import *
from figaro.metropolis import sample_point

from numba import jit, njit
from numba.extending import get_cython_function_address
import ctypes

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

#FIXME: jit
def compute_t_pars_array(k, mu, nu, L, samples, N, dim):
    # Compute hyperparameters
    k_n, mu_n, nu_n, L_n = compute_hyperpars_array(k, mu, nu, L, samples, N)
    # Update t-parameters
    t_df    = nu_n - dim + 1
    t_shape = L_n*(k_n+1)/(k_n*t_df)
    return t_df, t_shape, mu_n
    
def compute_hyperpars_array(k, mu, nu, L, samples, N):
    n_draws = samples.shape[0]
    means = np.mean(samples, axis = 1)
    if samples.shape[1] > 1:
        covs  = np.array([np.atleast_2d(np.cov(s, rowvar = False)) for s in samples]) #FIXME: is there a faster way?
    else:
        covs  = np.zeros(shape = (n_draws, samples.shape[2], samples.shape[2])) # 1 event: can't estimate covariance
    return _hyperpars_vect(k, mu, nu, L, means, covs, N, n_draws)

@jit
def _hyperpars_vect(k, mu, nu, L, means, covs, N, n_draws):
    k_n  = (k + N)
    nu_n = (nu + N)
    mu_n = (np.atleast_2d(mu)*k + N*means)/k_n
    L_n  = L*k + covs*N + k*N*((means - mu).T@(means - mu))/k_n
    return k_n, mu_n, nu_n, L_n
    
#FIXME: jit
def student_t_array(df, t, mu, sigma, dim, len_t):
    logP = np.zeros(len_t)
    for i in range(len_t):
        logP[i] = student_t(df = df, t = np.atleast_2d(t[i]), mu = np.atleast_2d(mu[i]), sigma = sigma[i], dim = dim)
    return logP

def build_mean_cov(x, dim):
    mean  = np.atleast_2d(x[:dim])
    corr  = np.identity(dim)/2.
    corr[np.triu_indices(dim, 1)] = x[2*dim:]
    corr  = corr + corr.T
    sigma = np.identity(dim)*x[dim:2*dim]**2
    cov_mat = sigma@corr
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
        self.w     = 0.

class component_h:
    def __init__(self, x, dim, prior):
        self.dim    = dim
        self.N      = 1
        self.events = [x]
        if self.dim == 1:
            sample = sample_point(self.events, a = 2, b = prior.L[0,0])
        else:
            integrator = cpnest.CPNest(Integrator(self.events, self.dim, prior.nu, prior.L),
                                            verbose = 0,
                                            nlive   = 200,
                                            maxmcmc = 5000,
                                            nensemble = 1,
                                            output = Path('.'),
                                            )
            integrator.run()
            sample = np.array(integrator.posterior_samples[-1].tolist())[:-2]
        self.mu, self.sigma = build_mean_cov(sample, self.dim)
        self.w      = 0.
    
class mixture:
    def __init__(self, means, covs, w, bounds, dim, n_cl):
        self.means  = means
        self.covs   = covs
        self.w      = w
        self.log_w  = np.log(w)
        self.bounds = bounds
        self.dim    = dim
        self.n_cl   = n_cl

    @probit
    def evaluate_mixture(self, x):
        p = np.sum(np.array([w*mn(mean, cov).pdf(x) for mean, cov, w in zip(self.means, self.covs, self.w)]), axis = 0)
        return p * np.exp(-probit_logJ(x, self.bounds))

    @probit
    def evaluate_log_mixture(self, x):
        p = logsumexp(np.array([w + mn(mean, cov).logpdf(x) for mean, cov, w in zip(self.means, self.covs, self.log_w)]), axis = 0)
        return p - probit_logJ(x, self.bounds)

    @from_probit
    def sample_from_dpgmm(self, n_samps):
        idx = np.random.choice(np.arange(self.n_cl), p = self.w, size = n_samps)
        ctr = Counter(idx)
        if self.dim > 1:
            samples = np.empty(shape = (1,self.dim))
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

class Integrator(cpnest.model.Model):
    
    def __init__(self, events, dim, df, L):
        super(Integrator, self).__init__()
        self.events    = events
        self.dim       = dim
        self.names     = ['m{0}'.format(i+1) for i in range(self.dim)] + ['s{0}'.format(i+1) for i in range(self.dim)] + ['r{0}'.format(j) for j in range(int(self.dim*(self.dim-1)/2.))]
        self.bounds    = [[-20, 20] for _ in range(self.dim)] + [[0, 1] for _ in range(self.dim)] + [[-1,1] for _ in range(int(self.dim*(self.dim-1)/2.))]
        self.prior     = invwishart(df = df, scale = L)
        self.mu_prior  = -self.dim*np.log(40)
    
    def log_prior(self, x):
        logP = super(Integrator, self).log_prior(x)
        if not np.isfinite(logP):
            return -np.inf
        self.mean, self.cov_mat = build_mean_cov(np.array(x.values), self.dim)
        if not np.linalg.slogdet(self.cov_mat)[0] > 0:
            return -np.inf
        logP = self.prior.logpdf(self.cov_mat) + self.mu_prior
        return logP
    
    def log_likelihood(self, x):
        return integrand(self.mean[0], self.cov_mat, self.events, self.dim)

#-------------------#
# Inference classes #
#-------------------#

class DPGMM:
    def __init__(self, bounds,
                       prior_pars = None,
                       alpha0     = 1.,
                       out_folder = '.'
                       ):
        self.bounds   = np.array(bounds)
        self.dim      = len(self.bounds)
        if prior_pars is not None:
            self.prior = prior(*prior_pars)
        else:
            self.prior = prior(1e-3, np.identity(self.dim)*0.3, self.dim, np.zeros(self.dim))
        self.alpha      = alpha0
        self.alpha_0    = alpha0
        self.mixture    = []
        self.n_cl       = 0
        self.n_pts      = 0
        self.normalised = False
    
    def initialise(self):
        self.alpha = self.alpha_0
        self.mixture  = []
        self.n_cl     = 0
        self.n_pts    = 0
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
            self.n_cl += 1
        else:
            self.mixture[int(cid)] = self.add_datapoint_to_component(x, self.mixture[int(cid)])
        return
    
    def density_from_samples(self, samples):
        for s in samples:
            self.add_new_point(np.atleast_2d(s))
    
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
        p = np.sum(np.array([w*mn(comp.mu, comp.sigma).pdf(x) for comp, w in zip(self.mixture, self.w)]), axis = 0)
        return p

    @probit
    def evaluate_mixture(self, x):
        self.normalise_mixture()
        p = np.sum(np.array([w*mn(comp.mu, comp.sigma).pdf(x) for comp, w in zip(self.mixture, self.w)]), axis = 0)
        return p
    
    @probit
    def evaluate_mixture_with_jacobian(self, x):
        self.normalise_mixture()
        p = np.sum(np.array([w*mn(comp.mu, comp.sigma).pdf(x) for comp, w in zip(self.mixture, self.w)]), axis = 0)
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
        
    def draw_sample(self):
        self.normalise_mixture()
        return mixture(np.array([comp.mu for comp in self.mixture]), np.array([comp.sigma for comp in self.mixture]), np.array(self.w), self.bounds, self.dim, self.n_cl)


class HDPGMM(DPGMM):
    def __init__(self, bounds,
                       alpha0     = 1.,
                       out_folder = '.',
                       prior_pars = None,
                       MC_draws   = 1e3,
                       ):
        dim = len(bounds)
        if prior_pars == None:
            prior_pars = (1e-3, np.identity(dim)*0.1, dim, np.zeros(dim))
        super().__init__(bounds = bounds, prior_pars = prior_pars, alpha0 = alpha0, out_folder = out_folder)
        self.MC_draws = int(MC_draws)
    
    def add_new_point(self, ev):
        self.n_pts += 1
        x = np.random.choice(ev)
        self.assign_to_cluster(x)
        self.alpha = update_alpha(self.alpha, self.n_pts, self.n_cl)

    def cluster_assignment_distribution(self, x):
        scores       = {}
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
            self.mixture.append(component_h(x, self.dim, self.prior))
            self.n_cl += 1
        else:
            self.mixture[int(cid)] = self.add_datapoint_to_component(x, self.mixture[int(cid)])
        return

    def log_predictive_likelihood(self, x, ss):
        if ss == "new":
            ss = component(np.zeros(self.dim), prior = self.prior)
            events = []
            ss.N = 0.
        else:
            events = ss.events

        if ss.N == 0:
            samples = np.zeros(shape = (self.MC_draws, 1, self.dim))
        else: #FIXME: check samples shape (I need rows with one sample per event)
            samples = [] # np.empty(len(events), dtype = np.ndarray)
            for i, ev in enumerate(events):
                s = ev._sample_from_dpgmm_probit(self.MC_draws)
                samples.append(s)
            samples = np.array(samples)
            samples = np.transpose(samples, (1,0,2)) # each row contains one point per event

        x_samples = x._sample_from_dpgmm_probit(self.MC_draws)
        
        t_df, t_shape, mu_n = compute_t_pars_array(self.prior.k, self.prior.mu, self.prior.nu, self.prior.L, samples, ss.N, self.dim)

        logL = student_t_array(df = t_df, t = x_samples, mu = mu_n, sigma = t_shape, dim = self.dim, len_t = self.MC_draws)
        logL = logsumexp(logL) - np.log(self.MC_draws)
        
        return logL

    def add_datapoint_to_component(self, x, ss):
        ss.events.append(x)
        if self.dim == 1:
            sample = sample_point(ss.events, a = 2, b = self.prior.L[0,0])
        else:
            integrator = cpnest.CPNest(Integrator(ss.events, self.dim, self.prior.nu, self.prior.L),
                                            verbose = 0,
                                            nlive   = 200,
                                            maxmcmc = 5000,
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
