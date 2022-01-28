import numpy as np

from scipy.special import gammaln
from scipy.stats import multivariate_normal as mn

import matplotlib.pyplot as plt
from matplotlib import rcParams
from corner import corner

from online_skyloc.decorators import *
from online_skyloc.coordinates import celestial_to_cartesian, cartesian_to_celestial

from pathlib import Path
from distutils.spawn import find_executable
from tqdm import tqdm

if find_executable('latex'):
    rcParams["text.usetex"] = True
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
        self.mean  = np.atleast_2d(x)
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

def student_t(df, t, mu, sigma, dim, s2max = np.inf):
    """
    http://gregorygundersen.com/blog/2020/01/20/multivariate-t/
    s2max can be removed if useless
    """
    vals, vecs = np.linalg.eigh(sigma)
    vals       = np.minimum(vals, s2max)
    logdet     = np.log(vals).sum()
    valsinv    = np.array([1./v for v in vals])
    U          = vecs * np.sqrt(valsinv)
    dev        = t - mu
    maha       = np.square(np.dot(dev, U)).sum(axis=-1)

    x = 0.5 * (df + dim)
    A = gammaln(x)
    B = gammaln(0.5 * df)
    C = dim/2. * np.log(df * np.pi)
    D = 0.5 * logdet
    E = -x * np.log1p((1./df) * maha)

    return float(A - B - C - D + E)

class mixture:
    def __init__(self, bounds,
                       prior_pars = None,
                       alpha0     = 1,
                       sigma_max  = 0.05,
                       ):
        self.bounds   = bounds
        self.dim      = len(self.bounds)
        if prior_pars is not None:
            self.prior = prior(*prior_pars)
        else:
            self.prior = prior(1, np.identity(self.dim)*0.5, self.dim, np.zeros(self.dim))
        self.alpha    = alpha0
        self.mixture  = []
        self.n_cl     = 0
        self.n_pts    = 0
        self.s2max    = (sigma_max)**2
        self.normalised = False

    def add_datapoint_to_component(self, x, ss):
        x = np.atleast_2d(x)
        old_mean = ss.mean
        ss.mean  = np.atleast_2d((ss.mean*(ss.N)+x)/(ss.N+1))
        ss.cov   = (ss.N*(ss.cov + np.matmul(old_mean.T, old_mean)) + np.matmul(x.T, x))/(ss.N+1) - np.matmul(ss.mean.T, ss.mean)
        ss.N     = ss.N+1
        ss.mu    = ((self.prior.mu*self.prior.k + ss.N*ss.mean)/(self.prior.k + ss.N))[0]
        ss.sigma = (np.atleast_2d(self.prior.L*self.prior.k) + ss.cov*ss.N + self.prior.k*ss.N*np.matmul((ss.mean - self.prior.mu).T, (ss.mean - self.prior.mu))/(self.prior.k + ss.N))/(self.prior.nu + ss.N)
        return ss

    def update_alpha(self, burnin = 200):
        a_old = self.alpha
        n     = self.n_pts
        K     = self.n_cl
        for _ in range(burnin+np.random.randint(100)):
            a_new = a_old + np.random.uniform(-1,1)*0.5
            if a_new > 0:
                logP_old = gammaln(a_old) - gammaln(a_old + n) + K * np.log(a_old) - 1./a_old
                logP_new = gammaln(a_new) - gammaln(a_new + n) + K * np.log(a_new) - 1./a_new
                if logP_new - logP_old > np.log(np.random.uniform()):
                    a_old = a_new
        self.alpha = a_old
        return

    def log_predictive_likelihood(self, x, ss):
        dim = x.shape[-1]
        if ss == "new":
            ss = component(np.zeros(dim), prior = self.prior)
            ss.N = 0
        mean = ss.mean
        S    = ss.cov
        N    = ss.N
        # Update hyperparameters
        k_n  = self.prior.k + N
        mu_n = np.atleast_2d((self.prior.mu*self.prior.k + N*mean)/k_n)
        nu_n = self.prior.nu + N
        L_n  = self.prior.L*self.prior.k + S*N + self.prior.k*N*np.matmul((mean - self.prior.mu).T, (mean - self.prior.mu))/k_n
        # Update t-parameters
        t_df    = nu_n - dim + 1
        t_shape = L_n*(k_n+1)/(k_n*t_df)
        # Compute logLikelihood
        logL = student_t(df = t_df, t = np.atleast_2d(x), mu = mu_n, sigma = t_shape, dim = dim, s2max = self.s2max)
        return logL

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
        self.update_alpha()
    
    def normalise_mixture(self):
        self.normalised = True
        for ss in self.mixture:
            ss.w = ss.N/self.n_pts
        self.mixture = np.array(self.mixture)
        self.w = np.array([ss.w for ss in self.mixture])
    
    @from_probit
    def sample_from_dpgmm(self, n_samps):
        if not self.normalised:
            self.normalise_mixture()
        idx = np.random.choice(np.arange(self.n_cl), p = self.w, size = n_samps)
        samples = np.array([mn(self.mixture[i].mu, self.mixture[i].sigma).rvs() for i in idx])
        return samples


class VolumeReconstruction(mixture):
    def __init__(self, max_dist,
                       out_folder   = '.',
                       prior_pars   = None,
                       alpha0       = 1,
                       sigma_max    = 0.05,
                       n_gridpoints = 100,
                       name         = 'skymap',
                       ):
        
        self.max_dist = max_dist
        bounds = np.array([[-max_dist, max_dist] for _ in range(3)])
        super().__init__(bounds, prior_pars, alpha0, sigma_max)
        
        self.n_gridpoints = n_gridpoints
        self.ra, self.dec, self.dist = np.meshgrid(np.linspace(0,2*np.pi, n_gridpoints), np.linspace(-np.pi/2., np.pi/2., n_gridpoints), np.linspace(1, max_dist, n_gridpoints))
        self.ra_2d, self.dec_2d = np.meshgrid(np.linspace(0,2*np.pi, n_gridpoints), np.linspace(-np.pi/2., np.pi/2., n_gridpoints))
        self.grid = np.transpose(np.array([self.ra, self.dec, self.dist]).reshape(3,-1))
        
        self.out_folder = Path(out_folder).resolve()
        self.skymap_folder = Path(out_folder, 'skymaps')
        if not self.skymap_folder.exists():
            self.skymap_folder.mkdir()
        self.name = name
        
    def add_sample(self, x):
        cart_x = celestial_to_cartesian(x)
        super().add_new_point(np.atleast_2d(x))
    
    def sample_from_volume(self, n_samps):
        samples = super().sample_from_dpgmm(n_samps)
        return cartesian_to_celestial(samples)
    
    def plot_samples(self, n_samps, initial_samples = None):
        mix_samples = self.sample_from_volume(n_samps)
        if initial_samples is not None:
            c = corner(initial_samples, color = 'orange', labels = ['$\\alpha$', '$\\delta$', '$D\ [Mpc]$'], hist_kwargs={'density':True})
            c = corner(mix_samples, fig = c, color = 'blue', labels = ['$\\alpha$', '$\\delta$', '$D\ [Mpc]$'], hist_kwargs={'density':True})
        else:
            c = corner(mix_samples, fig = c, color = 'blue', labels = ['$\\alpha$', '$\\delta$', '$D\ [Mpc]$'], hist_kwargs={'density':True})
        plt.savefig(Path(self.skymap_folder, 'samples'+self.name+'.pdf'), bbox_inches = 'tight')
    
    def evaluate_skymap(self):
        p_vol = np.zeros((self.n_gridpoints, self.n_gridpoints, self.n_gridpoints))
        for comp, w in zip(self.mixture, self.w):
            p_comp = w*mn(comp.mu, comp.sigma).pdf(self.grid).reshape(self.n_gridpoints, self.n_gridpoints, self.n_gridpoints)
            p_vol = p_vol + p_comp
        self.p_skymap = p_vol.sum(axis = -1)
    
    def make_skymap(self):
        self.evaluate_skymap()
        fig, ax = plt.subplots()
        ax.contourf(self.ra_2d, self.dec_2d, self.p_skymap, 1000)
        ax.set_xlabel('$\\alpha$')
        ax.set_ylabel('$\\delta$')
        fig.savefig(Path(self.skymap_folder, self.name+'.pdf'), bbox_inches = 'tight')
    
    def density_from_samples(self, samples):
        n_samps = len(samples)
        for s in tqdm(samples):
            self.add_sample(s)
        self.plot_samples(n_samps, initial_samples = samples)
        self.make_skymap()

        
