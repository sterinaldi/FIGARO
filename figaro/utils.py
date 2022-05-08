import numpy as np

import warnings
from pathlib import Path

from distutils.spawn import find_executable

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import axes
from matplotlib.projections import projection_registry
from corner import corner

from collections import Counter
import scipy.stats

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

class PPPlot(axes.Axes):
    """
    Construct a probability--probability (P--P) plot.

    Derived from https://lscsoft.docs.ligo.org/ligo.skymap/_modules/ligo/skymap/plot/pp.html#PPPlot
    This class avoids installing the whole ligo.skymap.plot package.
    """

    name = 'pp_plot'

    def __init__(self, *args, **kwargs):
        # Call parent constructor
        super().__init__(*args, **kwargs)

        # Square axes, limits from 0 to 1
        self.set_aspect(1.0)
        self.set_xlim(0.0, 1.0)
        self.set_ylim(0.0, 1.0)

    def add_diagonal(self, *args, **kwargs):
        """
        Add a diagonal line to the plot, running from (0, 0) to (1, 1).

        Other parameters
        ----------------
        kwargs :
            optional extra arguments to `matplotlib.axes.Axes.plot`
        """
        # Make copy of kwargs to pass to plot()
        kwargs = dict(kwargs)
        kwargs.setdefault('color', 'black')
        kwargs.setdefault('linestyle', 'dashed')
        kwargs.setdefault('linewidth', 0.5)

        # Plot diagonal line
        return self.plot([0, 1], [0, 1], *args, **kwargs)

    def add_confidence_band(self, nsamples, alpha=0.95, **kwargs):
        """
        Add a target confidence band.

        Parameters
        ----------
        nsamples : int
            Number of P-values
        alpha : float, default: 0.95
            Confidence level

        Other parameters
        ----------------
        **kwargs :
            optional extra arguments to `matplotlib.axes.Axes.fill_betweenx`
        """
        n = nsamples
        k = np.arange(0, n + 1)
        p = k / n
        ci_lo, ci_hi = scipy.stats.beta.interval(alpha, k + 1, n - k + 1)

        # Make copy of kwargs to pass to fill_betweenx()
        kwargs = dict(kwargs)
        kwargs.setdefault('color', 'ghostwhite')
        kwargs.setdefault('edgecolor', 'lightgray')
        kwargs.setdefault('linewidth', 0.5)
        fontsize = kwargs.pop('fontsize', 'x-small')

        return self.fill_betweenx(p, ci_lo, ci_hi, **kwargs)

    @classmethod
    def _as_mpl_axes(cls):
        """
        Support placement in figure using the `projection` keyword argument.
        See http://matplotlib.org/devel/add_new_projection.html.
        """
        return cls, {}
        
projection_registry.register(PPPlot)

def plot_median_cr(draws, injected = None, samples = None, selfunc = None, bounds = None, out_folder = '.', name = 'density', n_pts = 1000, label = None, unit = None, hierarchical = False, show = False, save = True, subfolder = False):
    """
    Plot the recovered 1D distribution along with the injected distribution and samples from the true distribution (both if available).
    
    Arguments:
        :iterable draws:                  container for mixture instances
        :callable or np.ndarray injected: injected distribution (if available)
        :np.ndarray samples:              samples from the true distribution (if available)
        :iterable bounds:                 bounds for the recovered distribution. If None, bounds from mixture instances are used.
        :str or Path out_folder:          output folder
        :str name:                        name to be given to outputs
        :int n_pts:                       number of points for linspace
        :str label:                       LaTeX-style quantity label, for plotting purposes
        :str unit:                        LaTeX-style quantity unit, for plotting purposes
        :bool hierarchical:               hierarchical inference, for plotting purposes
        :bool save:                       whether to save the plot or not
        :bool show:                       whether to show the plot during the run or not
    """
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
        ax.hist(samples, bins = int(np.sqrt(len(samples))), histtype = 'step', density = True, label = '$\mathrm{Samples}$', log = True)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
    else:
        ax.set_yscale('log')
    
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
        if selfunc is not None:
            if callable(selfunc):
                f_x = selfunc(x)
            else:
                f_x = selfunc
            filtered_p_x = p_x*f_x
            ax.plot(x, filtered_p_x/np.sum(filtered_p_x*dx), lw = 0.5, color = 'k', label = '$\mathrm{Selection\ effects}$')
        
    # Median
    ax.plot(x, p[50], lw = 0.7, color = 'steelblue', label = '${0}$'.format(rec_label))
    if label is None:
        label = 'x'
    if unit is None or unit == '':
        ax.set_xlabel('${0}$'.format(label))
    else:
        ax.set_xlabel('${0}\ [{1}]$'.format(label, unit))
    ax.set_ylabel('$p({0})$'.format(label))
    if samples is not None:
        ax.set_xlim(xlim)
    ax.set_ylim(bottom = 1e-5, top = np.max(p[95])*1.1)
    ax.grid(True,dashes=(1,3))
    ax.legend(loc = 0, frameon = False)
    if save:
        if subfolder:
            plot_folder = Path(out_folder, 'density')
            if not plot_folder.exists():
                plot_folder.mkdir()
            log_folder = Path(out_folder, 'log_density')
            if not log_folder.exists():
                log_folder.mkdir()
            txt_folder = Path(out_folder, 'txt')
            if not txt_folder.exists():
                txt_folder.mkdir()
        else:
            plot_folder = out_folder
            log_folder  = out_folder
            txt_folder  = out_folder
        fig.savefig(Path(log_folder, 'log_{0}.pdf'.format(name)), bbox_inches = 'tight')
        ax.set_yscale('linear')
        ax.autoscale(True)
        if samples is not None:
            ax.set_xlim(xlim)
        fig.savefig(Path(plot_folder, '{0}.pdf'.format(name)), bbox_inches = 'tight')
        np.savetxt(Path(txt_folder, 'prob_{0}.txt'.format(name)), np.array([x, p[50], p[5], p[16], p[84], p[95]]).T, header = 'x 50 5 16 84 95')
    if show:
        ax.set_yscale('linear')
        ax.autoscale(True)
        if samples is not None:
            ax.set_xlim(xlim)
        plt.show()
    plt.close()
    
    # If selection function is available, plot reweighted distribution
    if injected is not None and selfunc is not None:
        for perc in percentiles:
            p[perc] = np.percentile((probs/f_x).T, perc, axis = 1)
        norm = p[50].sum()*dx
        for perc in percentiles:
            p[perc] = p[perc]/norm
        
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        # CR
        ax.fill_between(x, p[95], p[5], color = 'mediumturquoise', alpha = 0.5)
        ax.fill_between(x, p[84], p[16], color = 'darkturquoise', alpha = 0.5)
        # Injection
        ax.plot(x, p_x, lw = 0.5, color = 'red', label = '$\mathrm{Simulated}$')
        # Median
        ax.plot(x, p[50], lw = 0.7, color = 'steelblue', label = '${0}$'.format(rec_label))
        if label is None:
            label = 'x'
        if unit is None or unit == '':
            ax.set_xlabel('${0}$'.format(label))
        else:
            ax.set_xlabel('${0}\ [{1}]$'.format(label, unit))
        ax.set_ylabel('$p({0})$'.format(label))
        ax.autoscale(True)
        ax.set_ylim(bottom = 1e-5, top = np.max(p[95])*1.1)
        ax.grid(True,dashes=(1,3))
        ax.legend(loc = 0, frameon = False)
        if save:
            fig.savefig(Path(log_folder, 'log_inj_{0}.pdf'.format(name)), bbox_inches = 'tight')
            ax.set_yscale('linear')
            ax.autoscale(True)
            fig.savefig(Path(plot_folder, 'inj_{0}.pdf'.format(name)), bbox_inches = 'tight')
            np.savetxt(Path(txt_folder, 'prob_inj_{0}.txt'.format(name)), np.array([x, p[50], p[5], p[16], p[84], p[95]]).T, header = 'x 50 5 16 84 95')
        if show:
            ax.set_yscale('linear')
            ax.autoscale(True)
            plt.show()
        plt.close()
    

def plot_multidim(draws, dim, samples = None, selfunc = None, out_folder = '.', name = 'density', labels = None, units = None, hierarchical = False, show = False, save = True, subfolder = False):
    """
    Plot the recovered multidimensional distribution along with samples from the true distribution (if available) as corner plot.
    
    Arguments:
        :iterable draws:         container for mixture instances
        :int dim:                number of dimensions
        :np.ndarray samples:     samples from the true distribution (if available)
        :str or Path out_folder: output folder
        :str name:               name to be given to outputs
        :list-of-str labels:     LaTeX-style quantity label, for plotting purposes
        :list-of-str units:      LaTeX-style quantity unit, for plotting purposes
        :bool hierarchical:      hierarchical inference, for plotting purposes
        :bool save:              whether to save the plot or not
        :bool show:              whether to show the plot during the run or not
    """
    if hierarchical:
        rec_label = '\mathrm{(H)DPGMM}'
    else:
        rec_label = '\mathrm{DPGMM}'
    
    if labels is None:
        labels = ['$x_{0}$'.format(i+1) for i in range(dim)]
    else:
        labels = ['${0}$'.format(l) for l in labels]
    
    if units is not None:
        labels = [l[:-1]+'\ [{0}]$'.format(u) if not u == '' else l for l, u in zip(labels, units)]
    
    # Draw samples from mixture
    if samples is not None:
        size = np.max([100**dim, len(samples)])
    else:
        size = 100**dim
        
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
        if not subfolder:
            c.savefig(Path(out_folder, '{0}.pdf'.format(name)), bbox_inches = 'tight')
        else:
            if not Path(out_folder, 'density').exists():
                Path(out_folder, 'density').mkdir()
            c.savefig(Path(out_folder, 'density', '{0}.pdf'.format(name)), bbox_inches = 'tight')
    plt.close()
    
    if selfunc is not None:
        return
        # Write importance sampler
        
        c = corner(mix_samples, color = 'dodgerblue', labels = labels, hist_kwargs={'density':True, 'label':'${0}$'.format(rec_label)})
        plt.legend(loc = 0, frameon = False, fontsize = 12, bbox_to_anchor = (0.95, (dim-1)+0.8))
        if show:
            plt.show()
        if save:
            if not subfolder:
                c.savefig(Path(out_folder, 'inj_{0}.pdf'.format(name)), bbox_inches = 'tight')
            else:
                c.savefig(Path(out_folder, 'density', 'inj_{0}.pdf'.format(name)), bbox_inches = 'tight')
        plt.close()

def plot_n_clusters_alpha(n_cl, alpha, out_folder = '.', name = 'event', show = False, save = True):
    """
    Plot the number of clusters and the concentration parameter as functions of the number of samples.
    
    Arguments:
        :np.ndarray n_cl:        number of active clusters
        :np.ndarray alpha:       concentration parameter
        :str or Path out_folder: output folder
        :str name:               name to be given to outputs
        :bool save:              whether to save the plot or not
        :bool show:              whether to show the plot during the run or not
    """
    fig, ax = plt.subplots()
    ax1 = ax.twinx()
    ax.plot(np.arange(1, len(n_cl)+1), n_cl, ls = '--', marker = '', lw = 0.7, color = 'k')
    ax1.plot(np.arange(1, len(alpha)+1), alpha, ls = '--', marker = '', lw = 0.7, color = 'r')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$N_{\mathrm{cl}}(t)$', color = 'k')
    ax1.set_ylabel('$\alpha(t)$', color = 'r')
    ax.grid(True,dashes=(1,3))
    if show:
        plt.show()
    if save:
        fig.savefig(Path(out_folder, '{0}_n_cl_alpha.pdf'.format(name)), bbox_inches = 'tight')
    plt.close()

def pp_plot_cdf(draws, injection, n_points = 1000, out_folder = '.', name = 'event', show = False, save = True):
    """
    Make pp-plot comparing draws cdfs and injection cdf
    
    Arguments:
        :iterable draws:         container of mixture instances
        :callable injection:     injected density
        :int n_points:           number of points for linspace
        :str or Path out_folder: output folder
        :str name:               name to be given to outputs
        :bool save:              whether to save the plot or not
        :bool show:              whether to show the plot during the run or not
    """
    all_bounds = np.atleast_2d([d.bounds[0] for d in draws])
    x_min = np.max(all_bounds[:,0])
    x_max = np.min(all_bounds[:,1])
    x = np.linspace(x_min, x_max, n_points+2)[1:-1]
    
    functions     = np.array([mix.evaluate_mixture(np.atleast_2d(x).T) for mix in draws])
    median        = np.percentile(functions.T, 50, axis = 1)
    cdf_draws     = np.array([fast_cumulative(d) for d in functions])
    cdf_median    = fast_cumulative(median)
    cdf_injection = fast_cumulative(injection(x))
    
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection = 'pp_plot')
    ax.add_confidence_band(len(cdf_median), alpha=0.95, color = 'ghostwhite')
    ax.add_diagonal()
    for cdf in cdf_draws:
        ax.plot(cdf_injection, cdf, lw = 0.5, alpha = 0.5, color = 'darkturquoise')
    ax.plot(cdf_injection, cdf, color = 'steelblue', lw = 0.7)
    ax.set_xlabel('$\mathrm{Injected}$')
    ax.set_ylabel('$\mathrm{FIGARO}$')
    ax.grid(True,dashes=(1,3))
    if show:
        plt.show()
    if save:
        fig.savefig(Path(out_folder, '{0}_ppplot.pdf'.format(name)), bbox_inches = 'tight')
    plt.close()
