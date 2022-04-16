import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from distutils.spawn import find_executable
from matplotlib import rcParams

if find_executable('latex'):
    rcParams["text.usetex"] = True
rcParams["xtick.labelsize"]=14
rcParams["ytick.labelsize"]=14
rcParams["xtick.direction"]="in"
rcParams["ytick.direction"]="in"
rcParams["legend.fontsize"]=12
rcParams["axes.labelsize"]=16
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.6

def plot_median_cr(draws, injected = None, samples = None, bounds = None, out_folder = '.', name = 'density', n_pts = 1000, label = 'x', unit = None, hierarchical = False, show = False, save = True):
    
    if hierarchical:
        rec_label = '\mathrm{(H)DPGMM}'
    else:
        rec_label = '\mathrm{DPGMM}'
    
    all_bounds = np.atleast_2d([d.bounds[0] for d in draws])
    x_min = np.max(all_bounds[:,0])
    x_max = np.min(all_bounds[:,1])
    
    if x_min == 0:
        x_min = 0.001
    if x_max == 0:
        x_max = -0.001
    
    if bounds is not None:
        if not bounds[0] >= x_min:
            warnings.warn("The provided lower bound is invalid for at least one draw. {0} will be used instead.".format(x_min))
        else:
            x_min = bounds[0]
        if not bounds[1] <= x_max:
            warnings.warn("The provided upper bound is invalid for at least one draw. {0} will be used instead.".format(x_max))
        else:
            x_max = bounds[1]
    
    x    = np.linspace(x_min, x_max, n_pts+2)[1:-1]
    dx   = x[1]-x[0]
    x_2d = np.atleast_2d(x).T
    
    probs = np.array([d.evaluate_mixture(x_2d) for d in draws])
    
    percentiles = [50, 5, 16, 84, 95]
    p = {}
    for perc in percentiles:
        p[perc] = np.percentile(probs.T, perc, axis = 1)
    norm = p[50].sum()*dx
    for perc in percentiles:
        p[perc] = p[perc]/norm
    
    fig, ax = plt.subplots()
    
    # Samples (if available)
    if samples is not None:
        ax.hist(samples, bins = int(np.sqrt(len(samples))), histtype = 'step', density = True, stacked = True, label = '$\mathrm{Samples}$')
    
    # CR
    ax.fill_between(x, p[95], p[5], color = 'mediumturquoise', alpha = 0.5)
    ax.fill_between(x, p[84], p[16], color = 'darkturquoise', alpha = 0.5)
    
    # Injection (if available)
    if injected is not None:
        if callable(injected):
            p_x = injected(x)
        else:
            p_x = injected
        ax.plot(x, p_x, lw = 0.5, color = 'red', label = '$\mathrm{Simulated}$')
    
    # Median
    ax.plot(x, p[50], lw = 0.5, color = 'steelblue', label = '${0}$'.format(rec_label))
    
    if unit is None:
        ax.set_xlabel('${0}$'.format(label))
    else:
        ax.set_xlabel('${0}\ [{1}]$'.format(label, unit))
    ax.set_ylabel('$p({0})$'.format(label))
    ax.grid(True,dashes=(1,3))
    ax.legend(loc = 0, frameon = False)
    if show:
        plt.show()
    if save:
        fig.savefig(Path(out_folder, '{0}.pdf'.format(name)), bbox_inches = 'tight')
        ax.set_yscale('log')
        fig.savefig(Path(out_folder, 'log_{0}.pdf'.format(name)), bbox_inches = 'tight')
        plt.close()
        np.savetxt(Path(out_folder, 'prob_{0}.txt'.format(name)), np.array([x, p[50], p[5], p[16], p[84], p[95]]).T, header = 'x 50 5 16 84 95')
