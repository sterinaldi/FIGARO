import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
from pathlib import Path
from figaro.cumulative import fast_cumulative
import ligo.skymap.plot

log2e = np.log2(np.e)

def angular_coefficient(x, y):
    return np.sum((x - np.mean(x))*(y - np.mean(y)))/np.sum((x - np.mean(x))**2)
    
def compute_angular_coefficients(x, step = 500):
    logN = np.arange(len(x))+1
    a    = np.zeros(len(x) - int(step), dtype = np.float64)
    for i in range(len(a)):
        a[i] = angular_coefficient(N[i:i+step], x[i:i+step])
    return a

@jit
def compute_autocorrelation(draws, mean, dx):
    taumax          = draws.shape[0]//2
    n_draws         = draws.shape[0]
    autocorrelation = np.zeros(taumax, dtype = np.float64)
    N = 0.
    for i in prange(n_draws):
        N += np.sum((draws[i] - mean)*(draws[i] - mean))*dx/n_draws
        
    for tau in prange(taumax):
        sum = 0.
        for i in prange(n_draws):
            sum += np.sum((draws[i] - mean)*(self.prob_draws[(i+tau)%n_draws] - mean))*dx/(N*n_draws)
        autocorrelation[tau] = sum
    return taumax, autocorrelation

def compute_entropy_single_draw(mixture, n_draws = 1e3):
    samples = mixture.sample_from_dpgmm(int(n_draws))
    logP    = mixture.evaluate_log_mixture(samples)
    entropy = np.sum(-logP)/(n_draws*log2e)
    return entropy

def compute_entropy(draws, n_draws = 1e3):
    S = np.zeros(len(draws))
    for i, d in enumerate(draws):
        S[i] = compute_entropy_single_draw(d, int(n_draws))
    return S

def autocorrelation(draws, xmin, xmax, out_folder, name = 'event', n_points = 1000):
    # 1-d only
    x  = np.linspace(xmin, xmax, n_points)
    dx = x[1] - x[0]
    
    functions = np.array([mix.evaluate_mixture(x) for mix in draws])
    mean      = np.mean(functions, axis = 0)
    
    taumax, ac = compute_autocorrelation(functions, mean, dx)
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(taumax), ac, ls = '--', marker = '', lw = 0.7)
    ax.set_xlabel('$\tau$')
    ax.set_ylabel('$C(\tau)$')
    ax.grid()
    fig.savefig(Path(out_folder, name+'_autocorrelation.pdf'), bbox_inches = 'tight')
    plt.close()

def entropy(draws, out_folder, name = 'event', n_draws = 1e3, step = 1, dim = 1):
    S = compute_entropy(draws, int(n_draws**dim))
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(draws)+1)*step, S, ls = '--', marker = '', lw = 0.7)
    ax.set_xlabel('$N$')
    ax.set_ylabel('$S(N)\ [\mathrm{bits}]$')
    ax.grid()
    fig.savefig(Path(out_folder, name+'_entropy.pdf'), bbox_inches = 'tight')
    plt.close()

def plot_n_clusters_alpha(n_cl, alpha, out_folder, name = 'event'):
    
    fig, ax = plt.subplots()
    ax1 = ax.twinx()
    ax.plot(np.arange(1, len(draws)+1), n_cl, ls = '--', marker = '', lw = 0.7, color = 'k')
    ax1.plot(np.arange(1, len(draws)+1), alpha, ls = '--', marker = '', lw = 0.7, color = 'r')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$N_{\mathrm{cl}}(t)$', color = 'k')
    ax1.set_ylabel('$\alpha(t)$', color = 'r')
    ax.grid()
    fig.savefig(Path(out_folder, name+'_n_cl_alpha.pdf'), bbox_inches = 'tight')
    plt.close()

def compute_entropy_rate_single_draw(mixture, n_draws = 1e3):
    samples = mixture._sample_from_dpgmm_probit(int(n_draws))
    logP    = mixture._evaluate_log_mixture_in_probit(samples)
    entropy = np.sum(-logP)/(n_draws*mixture.n_pts*log2e)
    return entropy

def compute_entropy_rate(draws, n_draws = 1e3):
    S = np.zeros(len(draws))
    for i, d in enumerate(draws):
        S[i] = compute_entropy_rate_single_draw(d, int(n_draws))
    return S

def entropy_rate(draws, out_folder, name = 'event', n_draws = 1e3, step = 1, dim = 1):
    S = compute_entropy(draws, int(n_draws**dim))
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(draws)+1)*step, S, ls = '--', marker = '', lw = 0.7)
    ax.set_xlabel('$N$')
    ax.set_ylabel('$R_S(N)\ [\mathrm{bits/sample}]$')
    ax.grid()
    fig.savefig(Path(out_folder, name+'_entropy.pdf'), bbox_inches = 'tight')
    plt.close()

def pp_plot(draws, injection, out_folder, name = 'event'):
    median        = np.percentile(draws.T, 50, axis = 1)
    cdf_draws     = np.array([fast_cumulative(d) for d in draws])
    cdf_median    = fast_cumulative(median)
    cdf_injection = fast_cumulative(injection)
    
    fig= plt.figure()
    ax = fig.add_subplot(111, projection = 'pp_plot')
    ax.add_confidence_band(len(cdf_median), alpha=0.95, color = 'paleturquoise')
    ax.add_diagonal()
    for cdf in cdf_draws:
        ax.plot(cdf_injection, cdf, lw = 0.5, alpha = 0.5, color = 'darkturquoise')
    ax.plot(cdf_injection, cdf, color = 'steelblue', lw = 0.7)
    ax.set_xlabel('$\mathrm{Injected}$')
    ax.set_ylabel('$\mathrm{FIGARO}$')
    ax.grid()
    fig.savefig(Path(out_folder, name+'_ppplot.pdf'), bbox_inches = 'tight')
    plt.close()
