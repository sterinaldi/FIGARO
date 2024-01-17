import numpy as np

from numba import njit, prange
from pathlib import Path

import matplotlib.pyplot as plt
from distutils.spawn import find_executable

from figaro.cumulative import fast_cumulative
from figaro.exceptions import FIGAROException
from figaro import plot_settings

log2e = np.log2(np.e)

@njit
def angular_coefficient(x, y):
    """
    Angular coefficient obtained from linear regression.
    
    Arguments:
        np.ndarray x: independent variables
        np.ndarray y: dependent variables
    
    Returns:
        double: angular coefficient
    """
    return np.sum((x - np.mean(x))*(y - np.mean(y)))/np.sum((x - np.mean(x))**2)
    
def compute_angular_coefficients(x, L = None):
    """
    Given an array of points x, computes the angular coefficient for each adjacent chunk of length L.
    
    Arguments:
        np.ndarray x: array of points
        int L:        window length
    
    Returns:
        np.ndarray: array of angular coefficients
    """
    if L is None:
       L = len(x)//10
    
    if L > len(x):
        raise FIGAROException("L must be smaller than the number of points you have")
        
    L = np.min([int(L), len(x)])
    N = np.arange(len(x))+1
    a = np.zeros(len(x) - int(L), dtype = np.float64)
    for i in range(len(a)):
        a[i] = angular_coefficient(N[i:i+L], x[i:i+L])
    return a

def plot_angular_coefficient(entropy, L = 500, ac_expected = None, out_folder = '.', name = 'event', step = 1, show = False, save = True):
    """
    Compute entropy angular coefficient and produce the relevant plot
    
    Arguments:
        iterable entropy:       container of mixture instances
        int L:                  window lenght
        double ac_expected:     expected angular coefficient
        str or Path out_folder: output folder
        str name:               name to be given to outputs
        int step:               number of draws between entropy samples (if downsampled by some other method, for plotting purposes only)
        bool save:              whether to save the plot or not
        bool show:              whether to show the plot during the run or not
    
    Returns:
        np.ndarray: angular coefficients
    """
    S = compute_angular_coefficients(entropy, L = L)
    fig, ax = plt.subplots()
    if ac_expected is not None:
        ax.axhline(ac_expected, lw = 0.5, ls = '--', c = 'r')
    ax.plot(np.arange(len(S))*step+L, S, ls = '--', marker = '', color = 'steelblue', lw = 0.7)
    ax.set_ylabel('$\\frac{dS(N)}{dN}$')
    ax.set_xlabel('$N$')
    if show:
        plt.show()
    if save:
        fig.savefig(Path(out_folder, name+'_angular_coefficient.pdf'), bbox_inches = 'tight')
    plt.close()
    return S

@njit
def compute_autocorrelation(draws, mean, dx):
    """
    Computes autocorrelation of subsequent draws as <∫(draw[i]-mean)*(draw[i+t]-mean)*dx/∫(draw[i]-mean)**2*dx>
    
    Arguments:
        np.ndarray draws: evaluated mixtures (2d array)
        np.ndarray mean:  bin-wise mean of evaluated mixtures (1d array)
        double dx:        integration measure
    
    Returns:
        int: upper bound of autocorrelation length
        np.ndarray: autocorrelation function
    """
    taumax          = draws.shape[0]//2
    n_draws         = draws.shape[0]
    autocorrelation = np.zeros(taumax, dtype = np.float64)
    N = 0.
    for i in prange(n_draws):
        N += np.sum((draws[i] - mean)*(draws[i] - mean))*dx/n_draws
        
    for tau in prange(taumax):
        sum = 0.
        for i in prange(n_draws):
            sum += np.sum((draws[i] - mean)*(draws[(i+tau)%n_draws] - mean))*dx/(N*n_draws)
        autocorrelation[tau] = sum
    return taumax, autocorrelation

def autocorrelation(draws, bounds = None, out_folder = '.', name = 'event', n_points = 1000, save = True, show = False):
    """
    Compute autocorrelation of a set of draws and produce the relevant plot
    
    Arguments:
        iterable draws:         container of mixture instances
        iterable bounds:        bounds of the interval over which the distributions are evaluated. It has to be passed as [[xmin,xmax]]. If None, draws bounds are used.
        str or Path out_folder: output folder
        str name:               name to be given to outputs
        int n_points:           number of points for linspace
        bool save:              whether to save the plot or not
        bool show:              whether to show the plot during the run or not
    
    Returns:
        np.ndarray: autocorrelation
    """
    all_bounds = np.atleast_2d([d.bounds[0] for d in draws])
    x_min = np.max(all_bounds[:,0])
    x_max = np.min(all_bounds[:,1])

    if bounds is not None:
        if not bounds[0] >= x_min:
            warnings.warn("The provided lower bound is invalid for at least one draw. {0} will be used instead.".format(x_min))
        else:
            x_min = bounds[0]
        if not bounds[1] <= x_max:
            warnings.warn("The provided upper bound is invalid for at least one draw. {0} will be used instead.".format(x_max))
        else:
            x_max = bounds[1]
    x  = np.linspace(x_min, x_max, n_points)
    dx = x[1] - x[0]
    
    functions = np.array([mix.pdf(x) for mix in draws])
    mean      = np.mean(functions, axis = 0)
    
    taumax, ac = compute_autocorrelation(functions, mean, dx)
    
    fig, ax = plt.subplots()
    ax.axhline(0, lw = 0.5, ls = '--', c = 'r')
    ax.plot(np.arange(taumax), ac, ls = '--', marker = '', lw = 0.7)
    ax.set_xlabel('$\\tau$')
    ax.set_ylabel('$C(\\tau)$')
    if show:
        plt.show()
    if save:
        fig.savefig(Path(out_folder, name+'_autocorrelation.pdf'), bbox_inches = 'tight')
    plt.close()
    return ac
    
def compute_entropy_single_draw(mixture, n_draws = 1e3, return_error = False):
    """
    Compute entropy for a single realisation of the DPGMM using Monte Carlo integration
    
    Arguments:
        mixture mixture: instance of mixture class (see mixture.py for definition)
        double n_draws:  number of MC draws
    
    Returns:
        double: entropy value
    """
    samples = mixture.rvs(int(n_draws))
    logP    = mixture.logpdf(samples)
    entropy = np.mean(-logP)*log2e
    if not return_error:
        return entropy
    else:
        dentropy = np.std(-logP)/(np.sqrt(n_draws)/log2e)
        return entropy, dentropy

def compute_entropy(draws, n_draws = 1e3, return_error = False):
    """
    Compute entropy for a list of realisations of the DPGMM using Monte Carlo integration
    
    Arguments:
        iterable draws: container of mixture class instaces (see mixture.py for definition)
        double n_draws: number of MC draws
    
    Returns:
        np.ndarray: entropy values
    """
    S = np.zeros(len(draws))
    if not return_error:
        for i, d in enumerate(draws):
            S[i] = compute_entropy_single_draw(d, n_draws)
        return S
    else:
        dS = np.zeros(len(draws))
        for i, d in enumerate(draws):
            S[i], dS[i] = compute_entropy_single_draw(d, n_draws, return_error = return_error)
        return S, dS

def entropy(draws, out_folder = '.', exp_entropy = None, name = 'event', n_draws = 1e4, step = 1, show = False, save = True):
    """
    Compute entropy of a set of draws and produce the relevant plot
    
    Arguments:
        iterable draws:         container of mixture instances
        str or Path out_folder: output folder
        double exp_entropy:     expected value for entropy, expressed in bits
        str name:               name to be given to outputs
        int n_draws:            number of MC draws
        int step:               number of draws between entropy samples (if downsampled by some other method, for plotting purposes only)
        bool save:              whether to save the plot or not
        bool show:              whether to show the plot during the run or not
    
    Return:
        np.ndarray: entropy
    """
    S = compute_entropy(draws, int(n_draws))
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(draws)+1)*step, S, ls = '--', marker = '', lw = 0.7)
    if exp_entropy is not None:
        ax.axhline(exp_entropy, lw = 0.5, ls = '--', c = 'r')
    ax.set_xlabel('$N$')
    ax.set_ylabel('$S(N)\ [\mathrm{bits}]$')
    if show:
        plt.show()
    if save:
        fig.savefig(Path(out_folder, name+'_entropy.pdf'), bbox_inches = 'tight')
    plt.close()
    return S
