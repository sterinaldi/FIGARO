import numpy as np
import warnings
import dill
import configparser

from pathlib import Path
from tqdm import tqdm
from typing import cast
from collections import Counter
from scipy.stats import multivariate_normal as mn

from figaro._numba_functions import *
from figaro.transform import transform_to_probit, transform_from_probit
from figaro.exceptions import FIGAROException

def recursive_grid(bounds, n_pts, get_1d = False):
    """
    Recursively generates the n-dimensional grid points (extremes are excluded).
    
    Arguments:
        list-of-lists bounds: extremes for each dimension (excluded)
        int n_pts:            number of points for each dimension
        bool get_1d:          return list of 1d-arrays (one per dimension)
        
    Returns:
        np.ndarray: grid
        np.ndarray: differential for each grid
        np.ndarray: list of 1d-arrays (one per dimension)
    """
    bounds = np.atleast_2d(bounds)
    n_pts  = np.atleast_1d(n_pts)
    if len(bounds) == 1:
        d  = np.linspace(bounds[0,0], bounds[0,1], n_pts[0])
        dD = d[1]-d[0]
        if get_1d:
            return np.atleast_2d(d).T, [dD], [d]
        return np.atleast_2d(d).T, [dD]
    else:
        if get_1d:
            grid_nm1, diff, l_1d = recursive_grid(np.array(bounds)[1:], n_pts[1:], get_1d)
        else:
            grid_nm1, diff = recursive_grid(np.array(bounds)[1:], n_pts[1:], get_1d)
        d = np.linspace(bounds[0,0], bounds[0,1], n_pts[0])
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
    Rejection sampler, allows for a selection function
    
    Arguments:
        int n_draws:      number of draws
        callable f:       probability density to sample from
        iterable bounds:  upper and lower bound
        callable selfunc: selection function, must support numpy arrays
    
    Returns:
        np.ndarray: samples
    """
    n_draws = int(n_draws)
    bounds  = np.atleast_2d(bounds)
    dim     = len(bounds)
    if selfunc is None:
        selfunc = lambda x: 1
    if dim == 1:
        pts = np.linspace(bounds[0,0], bounds[0,1], 1000)
    else:
        pts = np.random.uniform(bounds[:,0], bounds[:,1], size = (3*n_draws,len(bounds)))
    top = np.max(f(pts)*selfunc(pts))
    samples = []
    while len(samples) < n_draws:
        pts   = np.random.uniform(bounds[:,0], bounds[:,1], size = (n_draws,dim))
        if dim == 1:
            pts = pts.flatten()
        probs = f(pts)*selfunc(pts)
        h     = np.random.uniform(0, top, size = n_draws)
        samples.extend(pts[np.where(h < probs)])
    return np.array(samples)[:n_draws]

def get_priors(bounds, samples = None, mean = None, std = None, df = None, k = None, a = None, scale = None, probit = True, hierarchical = False):
    """
    This method takes the prior parameters for the Normal-Inverse-Wishart distribution in the natural space and returns them as parameters in the probit space, ordered as required by FIGARO. In the following, D will denote the dimensionality of the inferred distribution.

    Four parameters are returned:
        * df, is the number of degrees of freedom for the Inverse Wishart distribution,. It must be greater than D+1. If this parameter is None or does not satisfy the condition df > D+1, the default value D+2 is used;
        * k is the scale parameter for the multivariate Normal distribution. Suggested values are  k <~ 1e-1. If None, the default value 1e-2 is used.
        * mu is the mean of the multivariate Normal distribution. It can be either estimated from the available samples or passed directly as a 1D array with length D (the keyword argument mean overrides the samples). If None, the default value 0 (corresponding to the parameter space center) is used.
        * L is the expected value for the Inverse Wishart distribution. This parameter can be either:
            * passed as 1D array with shape (D,) or double: vector of standard deviations (if double, it assumes that the same std has to be used for all dimensions) - keyword std;
            * estimated from samples - keyword samples.
       
    The order in which they are returned is (k,L,df,mu).
    
    Arguments:
        np.ndarray bounds:              boundaries for probit transformation
        np.ndarray samples:             2D [DPGMM] or 3D [(H)DPGMM] array with samples
        double or np.ndarray mean:      mean [DPGMM]
        double or np.ndarray std:       expected standard deviation (if double, the same std is used for all dimensions, if np.ndarray must match the number of dimensions) [DPGMM and (H)DPGMM]
        int df:                         degrees of freedom for Inverse Wishart distribution [DPGMM]
        double k:                       scale parameter for Normal distribution [DPGMM]
        double a:                       shape parameter for the Inverse Gamma distribution [(H)DPGMM]
        double scale:                   fraction of samples std [DPGMM]
        bool probit:                    whether the probit transformation will be applied or not
        bool hierarchical:              returns the prior pars for (H)DPGMM rather than for DPGMM
        
    Returns:
        tuple: prior parameters ordered as in (H)/DPGMM
    """
    bounds = np.atleast_2d(bounds)
    dim = len(bounds)
    if scale is None:
        scale = 5.
    if samples is not None:
        if not np.iterable(samples[0]):
            samples = np.atleast_2d(samples).T
        if probit:
            if hierarchical:
                probit_samples = [transform_to_probit(s, bounds) for s in samples]
            else:
                probit_samples = transform_to_probit(samples, bounds)
    if hierarchical:
        if std is not None:
            out_sigma = np.atleast_2d(std)*np.ones((1, dim))
            if probit:
                out_sigma = transform_to_probit(np.mean(bounds, axis = -1)+out_sigma, bounds)
        elif samples is not None:
            if probit:
                all_samples     = np.concatenate(probit_samples, axis = 0)
                events_avg_cov  = np.diag(np.atleast_2d(np.mean([np.cov(ev.T) for ev in probit_samples], axis = 0)))
            else:
                all_samples     = np.concatenate(samples, axis = 0)
                events_avg_cov  = np.diag(np.atleast_2d(np.mean([np.cov(ev.T) for ev in samples], axis = 0)))
            all_samples_cov = np.diag(np.atleast_2d(np.cov(all_samples.T)))
            out_sigma       = (np.sqrt(all_samples_cov - events_avg_cov)/scale).flatten()
        else:
            out_sigma = np.diff(bounds, axis = -1)/scale
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
        if std is not None:
            L_out = np.identity(dim)*std**2
        elif samples is not None:
            if probit:
                cov_samples = np.atleast_2d(np.cov(probit_samples.T))
            else:
                cov_samples = np.atleast_2d(np.cov(samples.T))
            L_out = rescale_matrix(cov_samples, scale**2)
        else:
            if probit:
                sigma = transform_to_probit(np.atleast_2d(np.mean(bounds, axis = -1)+np.diff(bounds, axis = -1).flatten()/scale), bounds)[0]
                L_out = np.identity(dim)*sigma**2
            else:
                L_out = np.identity(dim)*(np.diff(bounds, axis = -1).flatten()/scale)**2
        # k
        if k is not None:
            k_out = k
        else:
            if samples is not None:
                if probit:
                    cov_samples = np.atleast_2d(np.cov(probit_samples.T))
                else:
                    cov_samples = np.atleast_2d(np.cov(samples.T))
                k_out = np.min(np.diag(L_out)/np.diag(cov_samples))
            else:
                k_out = 1./(scale)

        return (k_out, L_out, df_out, mu_out)

def rvs_median(draws, size = 1):
    """
    Generates samples from median distribution of a set of draws.
    
    Arguments:
        iterable draws: container for mixture instances
        int size:    number of samples
    
    Returns:
        np.ndarray: samples
    """
    idx = np.random.choice(np.arange(len(draws)), size = int(size))
    ctr = Counter(idx)
    samples = np.empty(shape = (1, draws[0].dim))
    for i, n in zip(ctr.keys(), ctr.values()):
        samples = np.concatenate((samples, draws[i].rvs(n)))
    return samples[1:]
    
def make_gaussian_mixture(mu, cov, bounds, out_folder = '.', names = None, save = False, save_samples = False, n_samps = 3000, probit = True):
    """
    Builds mixtures composed of equally-weighted Gaussian distribution.
    WARNING: due to the probit coordinate change, a Gaussian distribution in the natural space does not correspond to a Gaussian distribution in the probit space.
    The resulting distributions in probit space, therefore, are just an approximation. This approximation holds for distributions which are far from boundaries.
    In general, a more robust (but slower) approach would be to draw samples from each original Gaussian distribution and to use them to make a hierarchical inference.
    
    Arguments:
        np.ndarray mu:     mean for each Gaussian distribution
        np.ndarray cov:    covariance matrix for each Gaussian distribution
        np.ndarray bounds: boundaries for probit transformation
        str out_folder:    output folder
        bool save:         whether to save the draws or not
        bool save_samples: whether to save the samples or not
        int n_samps:       number of samples to estimate mean and covariance in probit space
        bool probit        whether to use the probit transformation or not
    
    Returns:
        np.ndarray: mixtures
    """
    # Here to avoid circular import
    from figaro.mixture import mixture
    bounds = np.atleast_2d(bounds)
    dim    = len(bounds)
    
    out_folder = Path(out_folder)
    if not out_folder.exists():
        out_folder.mkdir()
    
    if save:
        draws_folder = Path(out_folder, 'draws')
        if not draws_folder.exists():
            draws_folder.mkdir()
    
    if save_samples:
        events_folder = Path(out_folder, 'events')
        if not events_folder.exists():
            events_folder.mkdir()
    
    mixtures = []
    for i, (means, covs) in enumerate(zip(mu, cov)):
        means = np.atleast_2d(means).reshape(-1,dim)
        covs  = np.atleast_3d(covs).reshape(-1,dim,dim)
        if save_samples:
            samples = np.empty(shape = (1,len(bounds)))
        mm = []
        cc = []
        for m, c in zip(means, covs):
            if probit:
                ss = np.atleast_2d(mn(m, c, allow_singular = True).rvs(n_samps))
                # 1D issue
                if c.shape == (1,) or c.shape == (1,1):
                    ss = ss.T
                # Keeping only samples within bounds
                ss = ss[np.where((np.prod(bounds[:,0] < ss, axis = 1) & np.prod(ss < bounds[:,1], axis = 1)))]
                if save_samples:
                    samples = np.concatenate((samples, ss))
                # Probit samples
                p_ss = transform_to_probit(ss, bounds)
                mm.append(np.mean(p_ss, axis = 0))
                cc.append(np.atleast_2d(np.cov(p_ss.T)))
            else:
                mm.append(m)
                cc.append(c)
                if save:
                    ss = np.atleast_2d(mn(m, c, allow_singular = True).rvs(n_samps))
                    if ss.shape[0] == 1:
                        ss = ss.T
                    samples = np.concatenate((samples, ss))
        if save_samples:
            if names is not None:
                name = names[i]
            else:
                name = 'event_{0}'.format(i+1)
            np.savetxt(Path(events_folder, name+'.txt'), samples[1:])
        mix = mixture(np.atleast_2d(mm), np.atleast_3d(cc), np.ones(len(means))/len(means), bounds, len(bounds), len(means), None, probit = probit, alpha = 1.)
        if save:
            with open(Path(draws_folder, name+'.pkl'), 'wb') as f:
                dill.dump(np.array([mix]), f)
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
        obj options:            options
        str or Path out_folder: folder where to save the option file
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
        obj opts:         options object
        str or Path file: file with options
    
    Returns:
        obj: options
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
