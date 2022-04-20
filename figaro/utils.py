import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from distutils.spawn import find_executable
from matplotlib import rcParams
from corner import corner
from collections import Counter

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

#-------------#
#   Options   #
#-------------#

def is_opt_provided(parser, dest):
    """
    Checks if an option is provided by the user.
    From Oleg Gryb's answer in
    https://stackoverflow.com/questions/2593257/how-to-know-if-optparse-option-was-passed-in-the-command-line-or-as-a-default
    
    Arguments:
        :obj parser: an instance of optparse.OptionParser with the user-provided options
        :str dest:   name of the option
    Returns:
        :bool: True if the option is provided, false otherwise
    """
    for opt in parser._get_all_options():
        try:
            if opt.dest == dest and (opt._long_opts[0] in sys.argv[1:] or opt._short_opts[0] in sys.argv[1:]):
                return True
        except:
            if opt.dest == dest and opt._long_opts[0] in sys.argv[1:]:
                return True
    return False

def save_options(options):
    """
    Saves options for the run (reproducibility)
    
    Arguments:
        :dict options: options
    """
    logfile = open(Path(options.output, 'options_log.txt'), 'w')
    for key, val in zip(vars(options).keys(), vars(options).values()):
        logfile.write('{0}: {1}\n'.format(key,val))
    logfile.close()

#-------------#
#    Plots    #
#-------------#

def plot_median_cr(draws, injected = None, samples = None, bounds = None, out_folder = '.', name = 'density', n_pts = 1000, label = None, unit = None, hierarchical = False, show = False, save = True):
    
    if hierarchical:
        rec_label = '\mathrm{(H)DPGMM}'
    else:
        rec_label = '\mathrm{DPGMM}'
    
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
    ax.plot(x, p[50], lw = 0.7, color = 'steelblue', label = '${0}$'.format(rec_label))
    
    if label is None:
        label = 'x'
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

def plot_multidim(draws, dim, samples = None, out_folder = '.', name = 'density', labels = None, units = None, hierarchical = False, show = False, save = True):

    if hierarchical:
        rec_label = '\mathrm{(H)DPGMM}'
    else:
        rec_label = '\mathrm{DPGMM}'
    
    if labels is None:
        labels = ['$x_{0}$'.format(i+1) for i in range(dim)]
    else:
        labels = ['${0}$'.format(l) for l in labels]
    
    if units is not None:
        labels = [l[:-1]+'\ [{0}]$'.format(u) for l, u in zip(labels, units)]
    
    # Draw samples from mixture
    if samples is not None:
        size = np.min([1000, len(samples)])
    else:
        size = 1000
        
    idx = np.random.choice(np.arange(len(draws)), size = size)
    ctr = Counter(idx)
    
    mix_samples = np.empty(shape = (1,dim))
    for i, n in zip(ctr.keys(), ctr.values()):
        mix_samples = np.concatenate((mix_samples, draws[i].sample_from_dpgmm(n)))
    mix_samples = mix_samples[1:]
    
    # Make corner plots
    if samples is not None:
        c = corner(samples, color = 'coral', labels = labels, hist_kwargs={'density':True, 'label':'$\mathrm{Samples}$'})
        c = corner(mix_samples, fig = c, color = 'dodgerblue', labels = labels, hist_kwargs={'density':True, 'label':'${0}$'.format(rec_label)})
    else:
        c = corner(mix_samples, color = 'dodgerblue', labels = labels, hist_kwargs={'density':True, 'label':'${0}$'.format(rec_label)})
    plt.legend(loc = 0, frameon = False, fontsize = 12, bbox_to_anchor = (0.95, (dim-1)+0.8))
    if show:
        plt.show()
    if save:
        c.savefig(Path(out_folder, '{0}.pdf'.format(name)), bbox_inches = 'tight')
