import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange

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
    samples = mixture._sample_from_dpgmm_probit(int(n_draws))
    logP    = mixture._evaluate_log_mixture_in_probit(samples)
    entropy = np.sum(-logP)/n_draws
    return entropy

def compute_entropy(draws, n_draws = 1e3):
    S = np.zeros(len(draws))
    for i, d in enumerate(draws):
        S[i] = compute_entropy_single_draw(d, int(n_draws))
    return S

def autocorrelation(draws, xmin, xmax, out_folder, n_points = 1000):
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
    fig.savefig(Path(out_folder, 'autocorrelation.pdf'), bbox_inches = 'tight')
    plt.close()

def entropy(draws, out_folder, n_draws = 1e3):
    S = compute_entropy(draws, int(n_draws))
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(draws)+1), S, ls = '--', marker = '', lw = 0.7)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$S(t)$')
    ax.grid()
    fig.savefig(Path(out_folder, 'entropy.pdf'), bbox_inches = 'tight')
    plt.close()

def plot_n_clusters_alpha(n_cl, alpha, out_folder):
    
    fig, ax = plt.subplots()
    ax1 = ax.twinx()
    ax.plot(np.arange(1, len(draws)+1), n_cl, ls = '--', marker = '', lw = 0.7, color = 'k')
    ax1.plot(np.arange(1, len(draws)+1), alpha, ls = '--', marker = '', lw = 0.7, color = 'r')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$N_{\mathrm{cl}}(t)$', color = 'k')
    ax1.set_ylabel('$\alpha(t)$', color = 'r')
    ax.grid()
    fig.savefig(Path(out_folder, 'n_cl_alpha.pdf'), bbox_inches = 'tight')
    plt.close()
