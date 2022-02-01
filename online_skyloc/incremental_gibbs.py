import numpy as np
from itertools import product
from collections import Counter

from scipy.special import gammaln, logsumexp
from scipy.stats import multivariate_normal as mn

import matplotlib.pyplot as plt
from matplotlib import rcParams
from corner import corner

import dill

from online_skyloc.decorators import *
from online_skyloc.transform import *
from online_skyloc.coordinates import celestial_to_cartesian, cartesian_to_celestial, inv_Jacobian
from online_skyloc.credible_regions import ConfidenceArea

from pathlib import Path
from distutils.spawn import find_executable
from tqdm import tqdm

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

#if find_executable('latex'):
#    rcParams["text.usetex"] = True
rcParams["xtick.labelsize"]=14
rcParams["ytick.labelsize"]=14
rcParams["xtick.direction"]="in"
rcParams["ytick.direction"]="in"
rcParams["legend.fontsize"]=15
rcParams["axes.labelsize"]=16
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.6

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
                       ):
        self.bounds   = np.array(bounds)
        self.dim      = len(self.bounds)
        if prior_pars is not None:
            self.prior = prior(*prior_pars)
        else:
            self.prior = prior(1e-4, np.identity(self.dim)*0.5, self.dim, np.zeros(self.dim))
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
        samples = np.empty(shape = (1,3))
        for i, n in zip(ctr.keys(), ctr.values()):
            samples = np.concatenate((samples, np.atleast_2d(mn(self.mixture[i].mu, self.mixture[i].sigma).rvs(size = n))))
        return samples[1:]

    @probit
    def evaluate_mixture(self, x):
        self.normalise_mixture()
        p = np.zeros(len(x))
        for comp, w in zip(self.mixture, self.w):
            p += w*mn(comp.mu, comp.sigma).pdf(x)
        return p

    @probit
    def evaluate_log_mixture(self, x):
        self.normalise_mixture()
        p = np.ones(len(x)) * -np.inf
        for comp, w in zip(self.mixture, self.log_w):
            #FIXME: can be improved
            p = logsumexp((p, w + mn(comp.mu, comp.sigma).logpdf(x)), axis = 0)
        return p
        

class VolumeReconstruction(mixture):
    def __init__(self, max_dist,
                       out_folder   = '.',
                       prior_pars   = None,
                       alpha0       = 1,
                       sigma_max    = 0.05,
                       n_gridpoints = 100,
                       name         = 'skymap',
                       labels       = ['$\\alpha$', '$\\delta$', '$D\ [Mpc]$'],
                       levels       = [0.50, 0.90]
                       ):
        
        self.max_dist = max_dist
        bounds = np.array([[-max_dist, max_dist] for _ in range(3)])
        super().__init__(bounds, prior_pars, alpha0, sigma_max)
        
        self.n_gridpoints = n_gridpoints
        self.ra   = np.linspace(0,2*np.pi, 720)#n_gridpoints)
        self.dec  = np.linspace(-np.pi/2+0.01, np.pi/2.-0.01, 360)# n_gridpoints)
        self.dist = np.linspace(max_dist*0.01, max_dist*0.99, n_gridpoints)
        self.dD   = max_dist/n_gridpoints
        self.dra  = 2*np.pi/n_gridpoints
        self.ddec = np.pi/n_gridpoints
        self.grid = np.array([v for v in product(*(self.ra,self.dec,self.dist))])
        self.grid2d = np.array([v for v in product(*(self.ra,self.dec))])
        self.ra_2d, self.dec_2d = np.meshgrid(self.ra, self.dec)
        self.levels = levels
        
        self.out_folder = Path(out_folder).resolve()
        self.skymap_folder = Path(out_folder, 'skymaps')
        if not self.skymap_folder.exists():
            self.skymap_folder.mkdir()
        self.density_folder = Path(out_folder, 'density')
        if not self.density_folder.exists():
            self.density_folder.mkdir()
        self.name = name
        self.labels = labels
        
    def add_sample(self, x):
        cart_x = celestial_to_cartesian(x)
        self.add_new_point(cart_x)
        if self.n_pts % 100000 == 0:
            self.make_skymap()
    
    def sample_from_volume(self, n_samps):
        samples = self.sample_from_dpgmm(n_samps)
        return cartesian_to_celestial(samples)
    
    def plot_samples(self, n_samps, initial_samples = None):
        mix_samples = self.sample_from_volume(n_samps)
        if initial_samples is not None:
            c = corner(initial_samples, color = 'coral', labels = self.labels, hist_kwargs={'density':True, 'label':'$\mathrm{Samples}$'})
            c = corner(mix_samples, fig = c, color = 'dodgerblue', labels = self.labels, hist_kwargs={'density':True, 'label':'$\mathrm{DPGMM}$'})
        else:
            c = corner(mix_samples, fig = c, color = 'dodgerblue', labels = self.labels, hist_kwargs={'density':True, 'label':'$\mathrm{DPGMM}$'})
        plt.legend(loc = 0, frameon = False,fontsize = 15, bbox_to_anchor = (1-0.05, 2.8))
        plt.savefig(Path(self.skymap_folder, 'samples_'+self.name+'.pdf'), bbox_inches = 'tight')
        plt.close()
    
    def evaluate_skymap(self):
        
        cartesian_grid = celestial_to_cartesian(self.grid)
        
        log_inv_J     = -np.log(inv_Jacobian(self.grid)).reshape(len(self.ra), len(self.dec), len(self.dist)) + probit_logJ(transform_to_probit(cartesian_grid, self.bounds), self.bounds).reshape(len(self.ra), len(self.dec), len(self.dist))
        inv_J = np.exp(log_inv_J)
        
        p_vol     = self.evaluate_mixture(cartesian_grid).reshape(len(self.ra), len(self.dec), len(self.dist)) * inv_J
        log_p_vol = self.evaluate_log_mixture(cartesian_grid).reshape(len(self.ra), len(self.dec), len(self.dist)) + log_inv_J

        norm_p_vol = (p_vol*self.dD*self.dra*self.ddec).sum()
        
        p_vol      = p_vol/norm_p_vol
        log_p_vol  = log_p_vol - np.log(norm_p_vol)
        
        self.p_skymap = (p_vol*self.dD).sum(axis = -1)
        
        self.log_p_skymap = np.log(self.p_skymap)
#        self.log_p_skymap = logsumexp((log_p_vol + log_inv_J + np.log(self.dD)), axis = -1) #FIXME: check

        self.areas, self.idx_CR, self.heights = ConfidenceArea(self.log_p_skymap, self.dec, self.ra, adLevels = self.levels)
            
    def make_skymap(self):
        self.evaluate_skymap()
        fig = plt.figure()
        ax = fig.add_subplot(111)#, projection='mollweide')
        c = ax.contourf(self.ra_2d, self.dec_2d, self.p_skymap.T, 990, cmap = 'Reds')
        c1 = ax.contour(self.ra_2d, self.dec_2d, self.log_p_skymap.T, np.sort(self.heights), colors = 'black', linewidths = 0.7)
        ax.clabel(c1, fmt = {l:'{0:.0f}%'.format(100*s) for l,s in zip(c1.levels, self.levels[::-1])}, fontsize = 5)
        for i in range(len(self.areas)):
            c1.collections[i].set_label('${0:.0f}\\%'.format(100*self.levels[-i])+ '\ \mathrm{CR}:'+'{0:.1f}'.format(self.areas[-i]) + '\ \mathrm{deg}^2$')
        ax.set_xlabel('$\\alpha$')
        ax.set_ylabel('$\\delta$')
        ax.legend(loc = 0, frameon = False, fontsize = 10, handlelength=0, handletextpad=0)
        plt.colorbar(c, label = '$p(\\alpha,\\delta)$', orientation='horizontal')
        fig.savefig(Path(self.skymap_folder, self.name+'.pdf'), bbox_inches = 'tight')
        plt.close()
    
    def save_density(self):
        with open(Path(self.density_folder, self.name + '_density.pkl'), 'wb') as dill_file:
            dill.dump(self, dill_file)

    def density_from_samples(self, samples):
        n_samps = len(samples)
        samples_copy = np.copy(samples)
        for s in tqdm(samples_copy):
            self.add_sample(s)
        self.plot_samples(n_samps, initial_samples = samples)
        self.make_skymap()
        self.save_density()

    def evaluate_density(self, x):
        return self.evaluate_mixture(celestial_to_cartesian(x))
