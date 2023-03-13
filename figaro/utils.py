import numpy as np
import warnings
import dill
import configparser

from pathlib import Path
from tqdm import tqdm
from typing import cast
from collections import Counter
from scipy.stats import multivariate_normal as mn

from figaro.transform import transform_to_probit
from figaro.exceptions import FIGAROException

#-–––––––––-#
# Utilities #
#-----------#

def recursive_grid(bounds, n_pts, get_1d = False):
    """
    Recursively generates the n-dimensional grid points (extremes are excluded).
    
    Arguments:
        :list-of-lists bounds: extremes for each dimension (excluded)
        :int n_pts:            number of points for each dimension
        :bool get_1d:          return list of 1d-arrays (one per dimension)
        
    Returns:
        :np.ndarray: grid
        :np.ndarray: differential for each grid
        :np.ndarray: list of 1d-arrays (one per dimension)
    """
    bounds = np.atleast_2d(bounds)
    n_pts  = np.atleast_1d(n_pts)
    if len(bounds) == 1:
        d  = np.linspace(bounds[0,0], bounds[0,1], n_pts[0]+2)[1:-1]
        dD = d[1]-d[0]
        if get_1d:
            return np.atleast_2d(d).T, [dD], [d]
        return np.atleast_2d(d).T, [dD]
    else:
        if get_1d:
            grid_nm1, diff, l_1d = recursive_grid(np.array(bounds)[1:], n_pts[1:], get_1d)
        else:
            grid_nm1, diff = recursive_grid(np.array(bounds)[1:], n_pts[1:], get_1d)
        d = np.linspace(bounds[0,0], bounds[0,1], n_pts[0]+2)[1:-1]
        diff.insert(0, d[1]-d[0])
        grid     = []
        for di in d:
            for gi in grid_nm1:
                grid.append([di,*gi])
        if get_1d:
            l_1d.insert(0, d)
            return np.array(grid), diff, l_1d
        return np.array(grid), diff

def rejection_sampler(n_draws, f, bounds, selfunc = None):
    """
    1D rejection sampler, allows for a selection function
    
    Arguments:
        :int n_draws:      number of draws
        :callable f:       probability density to sample from
        :iterable bounds:  upper and lower bound
        :callable selfunc: selection function, must support numpy arrays
    
    Returns:
        :np.ndarray: samples
    """
    n_draws = int(n_draws)
    if selfunc is None:
        selfunc = lambda x: 1
    x   = np.linspace(bounds[0], bounds[1], 1000)
    top = np.max(f(x)*selfunc(x))
    samples = []
    while len(samples) < n_draws:
        pts   = np.random.uniform(bounds[0], bounds[1], size = n_draws)
        probs = f(pts)*selfunc(pts)
        h     = np.random.uniform(0, top, size = n_draws)
        samples.extend(pts[np.where(h < probs)])
    return np.array(samples).flatten()[:n_draws]

def get_priors(bounds, samples = None, mean = None, std = None, cov = None, df = None, k = None, a = None, scale = 5., probit = True, hierarchical = False):
    """
    This method takes the prior parameters for the Normal-Inverse-Wishart distribution in the natural space and returns them as parameters in the probit space, ordered as required by FIGARO. In the following, D will denote the dimensionality of the inferred distribution.

    Four parameters are returned:
    * df, is the number of degrees of freedom for the Inverse Wishart distribution,. It must be greater than D+1. If this parameter is None or does not satisfy the condition df > D+1, the default value D+2 is used;
    * k is the scale parameter for the multivariate Normal distribution. Suggested values are  k <~ 1e-1. If None, the default value 1e-2 is used.
    * mu is the mean of the multivariate Normal distribution. It can be either estimated from the available samples or passed directly as a 1D array with length D (the keyword argument mean overrides the samples). If None, the default value 0 (corresponding to the parameter space center) is used.
    * L is the expected value for the Inverse Wishart distribution. This parameter can be either (in descending priority order):
        * passed as 2D array with shape (D,D), the covariance matrix - keyword cov;
        * passed as 1D array with shape (D,) or double: vector of standard deviations (if double, it assumes that the same std has to be used for all dimensions) - keyword std;
        * estimated from samples - keyword samples.
       
    The order in which they are returned is (k,L,df,mu).
    
    Arguments:
        :np.ndarray bounds:              boundaries for probit transformation
        :np.ndarray samples:             2D array with samples
        :double or np.ndarray mean:      mean [DPGMM]
        :double or np.ndarray std:       expected standard deviation (if double, the same std is used for all dimensions, if np.ndarray must match the number of dimensions) [DPGMM and (H)DPGMM]
        :np.ndarray cov:                 covariance matrix [DPGMM]
        :int df:                         degrees of freedom for Inverse Wishart distribution [DPGMM]
        :double k:                       scale parameter for Normal distribution [DPGMM]
        :double a:                       shape parameter for the Inverse Gamma distribution [(H)DPGMM]
        :double scale:                   fraction of samples std [DPGMM]
        :bool probit:                    whether the probit transformation will be applied or not
        :bool hierarchical:              returns the prior pars for (H)DPGMM rather than for DPGMM
        
    Returns:
        :tuple: prior parameters ordered as in (H)/DPGMM
    """
    bounds = np.atleast_2d(bounds)
    dim = len(bounds)
    if samples is not None:
        if len(np.shape(samples)) < 2:
            samples = np.atleast_2d(samples).T
        if probit:
            probit_samples = transform_to_probit(samples, bounds)
    if hierarchical:
        if std is not None:
            out_sigma = np.atleast_2d(std)*np.ones((1, dim))
            if probit:
                out_sigma = transform_to_probit(np.mean(bounds, axis = -1)+out_sigma, bounds)
        elif samples is not None:
            if probit:
                out_sigma = np.sqrt(np.diag(np.atleast_2d(np.cov(probit_samples.T))))
            else:
                out_sigma = np.sqrt(np.diag(np.atleast_2d(np.cov(samples.T))))
            out_sigma = out_sigma/scale
        else:
            out_sigma = np.diff(bounds, axis = -1)/10.
            if probit:
                out_sigma = transform_to_probit(np.mean(bounds, axis = -1)+out_sigma, bounds)
        out_sigma = out_sigma.flatten()
        if a is not None:
            out_a = a
        else:
            out_a = 2.
            
        return (out_sigma, out_a)
    
    else:
        # DF
        if df is not None and df > dim+2:
            df_out = df
        else:
            df_out = dim+2
            
        draw_flag = False
        # Mu
        if mean is not None:
            mean = np.atleast_1d(mean)
            if not np.prod(bounds[:,0] < mean) & np.prod(mean < bounds[:,1]):
                raise ValueError("Mean is outside of the given bounds")
            if probit:
                mu_out = transform_to_probit(mean, bounds)
            else:
                mu_out = mean
        elif samples is not None:
            if probit:
                mu_out = np.atleast_1d(np.mean(probit_samples, axis = 0))
            else:
                mu_out = np.atleast_1d(np.mean(samples, axis = 0))
        else:
            if probit:
                mu_out = np.zeros(dim)
            else:
                mu_out = np.atleast_1d(np.mean(bounds, axis = 1))
        # L
        if cov is not None:
            L_out = cov
            if probit:
                draw_flag = True
        elif std is not None:
            L_out = np.identity(dim)*std**2
            if probit:
                draw_flag = True
        elif samples is not None:
            if probit:
                L_out = np.atleast_2d(np.cov(probit_samples.T))
            else:
                L_out = np.atleast_2d(np.cov(samples.T))
            L_out = np.identity(dim)*np.diag(L_out/scale**2)
        else:
            if probit:
                L_out = np.identity(dim)*0.2**2
            else:
                L_out = np.identity(dim)*(np.diff(bounds, axis = 1)/10)**2
        if draw_flag:
            ss = mn(np.mean(bounds, axis = -1), L_out).rvs(3000)
            if dim == 1:
                ss = np.atleast_2d(ss).T
            # Keeping only samples within bounds
            ss = ss[np.where((np.prod(bounds[:,0] < ss, axis = 1) & np.prod(ss < bounds[:,1], axis = 1)))]
            probit_samples = transform_to_probit(ss, bounds)
            L_out = np.atleast_2d(np.cov(probit_samples.T))
        # k
        if k is not None:
            k_out = k
        else:
            s, log_k_out = np.linalg.slogdet(L_out)
            k_out = np.exp(log_k_out/dim)
        return (k_out, L_out, df_out, mu_out)

def rvs_median(draws, n_draws):
    """
    Generates samples from median distribution of a set of draws.
    
    Arguments:
        :iterable draws: container for mixture instances
        :int n_draws:    number of samples
    
    Returns:
        :np.ndarray: samples
    """
    idx = np.random.choice(np.arange(len(draws)), size = int(n_draws))
    ctr = Counter(idx)
    samples = np.empty(shape = (1, draws[0].dim))
    for i, n in zip(ctr.keys(), ctr.values()):
        samples = np.concatenate((samples, draws[i].rvs(n)))
    return samples[1:]
    
def make_single_gaussian_mixture(mu, cov, bounds, out_folder = '.', save = False, n_samps = 3000, probit = True):
    """
    Builds mixtures composed of a single Gaussian distribution.
    WARNING: due to the probit coordinate change, a Gaussian distribution in the natural space does not correspond to a Gaussian distribution in the probit space.
    The resulting distributions, therefore, are just an approximation. This approximation holds for distributions which are far from boundaries.
    In general, a more robust (but slower) approach would be to draw samples from each original Gaussian distribution and to use them to make a hierarchical inference.
    
    Arguments:
        :np.ndarray mu:     mean for each Gaussian distribution
        :np.ndarray cov:    covariance matrix for each Gaussian distribution
        :np.ndarray bounds: boundaries for probit transformation
        :str out_folder:    output folder
        :bool save:         whether to save the draws or not
        :int n_samps:       number of samples to estimate mean and covariance in probit space
    
    Returns:
        :np.ndarray: mixtures
    """
    # Here to avoid circular import
    from figaro.mixture import mixture
    bounds = np.atleast_2d(bounds)
    
    out_folder = Path(out_folder)
    if not out_folder.exists():
        out_folder.mkdir()
    
    draws_folder = Path(out_folder, 'draws')
    if not draws_folder.exists():
        draws_folder.mkdir()
    
    events_folder = Path(out_folder, 'events')
    if not events_folder.exists():
        events_folder.mkdir()
    
    if len(cov.shape) == 1:
        cov = np.atleast_2d(cov).T
    
    mixtures = []
    for i, (m, c) in enumerate(zip(mu, cov)):
        if probit:
            ss = np.atleast_2d(mn(m, c).rvs(n_samps))
            # 1D issue
            if c.shape == (1,) or c.shape == (1,1):
                ss = ss.T
            # Keeping only samples within bounds
            ss = ss[np.where((np.prod(bounds[:,0] < ss, axis = 1) & np.prod(ss < bounds[:,1], axis = 1)))]
            if save:
                np.savetxt(Path(events_folder, 'event_{0}.txt'.format(i+1)), ss)
            # Probit samples
            p_ss = transform_to_probit(ss, bounds)
            mm = np.mean(p_ss, axis = 0)
            cc = np.atleast_2d(np.cov(p_ss.T))
        else:
            mm = m
            cc = c
            if save:
                ss = np.atleast_2d(mn(m, c).rvs(n_samps))
                if c.shape == (1,1):
                    ss = ss.T
                np.savetxt(Path(events_folder, 'event_{0}.txt'.format(i+1)), ss)
        mix = mixture(np.atleast_2d([mm]), np.atleast_3d([cc]), np.ones(1), bounds, len(bounds), 1, None, probit = probit)
        mixtures.append([mix])
    
    mixtures = np.array(mixtures)
    
    if save:
        with open(Path(draws_folder, 'posteriors_single_event.pkl'), 'wb') as f:
            dill.dump(mixtures, f)
    
    return mixtures

#-------------#
#   Options   #
#-------------#

def save_options(options, out_folder, name = None):
    """
    Saves options for the run (reproducibility)
    
    Arguments:
        :obj options:            options
        :str or Path out_folder: folder where to save the option file
    """
    if name is None:
        filename = 'options_log.ini'
    else:
        filename = 'options_log_{0}.ini'.format(name)
    with open(Path(out_folder, filename), 'w') as logfile:
        logfile.write('[OPTIONS]\n')
        for key, val in zip(vars(options).keys(), vars(options).values()):
            logfile.write('{0} = {1}\n'.format(key,val))

def load_options(opts, file):
    """
    Loads options for the run (reproducibility)
    
    Arguments:
        :obj opts:         options object
        :str or Path file: file with options
    
    Returns:
        :obj: options
    """
    with open(file, 'r') as logfile:
        config = configparser.RawConfigParser()
        config.read(file)
        opts_dict = dict(config.items('OPTIONS'))
    for key in opts_dict.keys():
        if opts_dict[key] in ['True', 'False']:
            if opts_dict[key] == 'True':
                opts_dict[key] = True
            if opts_dict[key] == 'False':
                opts_dict[key] = False
        if key == 'bounds':
            opts.bounds = str(opts_dict['bounds'])
        else:
            try:
                exec('opts.{0} = opts.{0}.__class__(opts_dict["{0}"])'.format(key))
            except TypeError:
                exec('opts.{0} = opts.{0}.__class__()'.format(key))
    return opts
