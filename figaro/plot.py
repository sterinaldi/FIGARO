import numpy as np
import warnings

from pathlib import Path
from scipy.stats import beta
from corner import corner

import matplotlib.pyplot as plt
from matplotlib import axes, colormaps
from matplotlib.projections import projection_registry
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from figaro import plot_settings
from figaro.marginal import marginalise
from figaro.credible_regions import ConfidenceArea
from figaro.utils import recursive_grid

# Telling python to ignore empty legend warning from matplotlib
warnings.filterwarnings("ignore", message = "No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.")

class PPPlot(axes.Axes):
    """
    Construct a probability-probability (P-P) plot.

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

    def add_confidence_band(self, nsamples, cl=0.9, **kwargs):
        """
        Add a target confidence band.

        Parameters
        ----------
        nsamples : int
            Number of P-values
        cl : float, default: 0.9
            Confidence level

        Other parameters
        ----------------
        **kwargs :
            optional extra arguments to `matplotlib.axes.Axes.fill_betweenx`
        """
        n = nsamples
        k = np.arange(0, n + 1)
        p = k / n
        ci_lo, ci_hi = beta.interval(cl, k + 1, n - k + 1)

        # Make copy of kwargs to pass to fill_betweenx()
        kwargs = dict(kwargs)
        kwargs.setdefault('color', 'ghostwhite')
        kwargs.setdefault('edgecolor', 'gray')
        kwargs.setdefault('linewidth', 0.5)
        kwargs.setdefault('alpha', 0.5)
        fontsize = kwargs.pop('fontsize', 'x-small')

        return self.fill_betweenx(p, ci_lo, ci_hi, **kwargs, label = '${0}\%\ CR$'.format(int(cl*100)))

    @classmethod
    def _as_mpl_axes(cls):
        """
        Support placement in figure using the `projection` keyword argument.
        See http://matplotlib.org/devel/add_new_projection.html.
        """
        return cls, {}
        
projection_registry.register(PPPlot)

def plot_median_cr(draws, injected = None, samples = None, selfunc = None, bounds = None, out_folder = '.', name = 'density', n_pts = 1000, label = None, unit = None, hierarchical = False, show = False, save = True, subfolder = False, true_value = None, true_value_label = '\mathrm{True\ value}', injected_label = '\mathrm{Simulated}', median_label = None, fig = None):
    """
    Plot the recovered 1D distribution along with the injected distribution and samples from the true distribution (both if available).
    
    Arguments:
        iterable draws:                  container for mixture instances
        callable or np.ndarray injected: injected distribution (if available)
        np.ndarray samples:              samples from the true distribution (if available)
        callable or np.ndarray selfunc:  selection function (if available)
        iterable bounds:                 bounds for the recovered distribution. If None, bounds from mixture instances are used.
        str or Path out_folder:          output folder
        str name:                        name to be given to outputs
        int n_pts:                       number of points for linspace
        str label:                       LaTeX-style quantity label, for plotting purposes
        str unit:                        LaTeX-style quantity unit, for plotting purposes
        bool hierarchical:               hierarchical inference, for plotting purposes
        bool save:                       whether to save the plots or not
        bool show:                       whether to show the plots during the run or not
        bool subfolder:                  whether to save the plots in different subfolders (for multiple events)
        float true_value:                true value to infer
        str true_value_label:            label to assign to the true value marker
        str injected_label:              label to assign to the injected distribution
        str median_label:                label to assign to the reconstruction
        matplotlib.figure.Figure fig:    figure to use for plotting. Must have (dim,dim) axes.
    
    Returns:
        matplotlib.figure.Figure: figure with the plot
    """
    if median_label is None:
        if hierarchical:
            median_label = '\mathrm{(H)DPGMM}'
        else:
            median_label = '\mathrm{DPGMM}'
    
    all_bounds = np.atleast_2d([d.bounds[0] for d in draws])
    x_min = np.max(all_bounds[:,0])
    x_max = np.min(all_bounds[:,1])
    
    probit = np.array([d.probit for d in draws]).any()
    
    if bounds is not None:
        if not bounds[0] >= x_min and probit:
            warnings.warn("The provided lower bound is invalid for at least one draw. {0} will be used instead.".format(x_min))
        else:
            x_min = bounds[0]
        if not bounds[1] <= x_max and probit:
            warnings.warn("The provided upper bound is invalid for at least one draw. {0} will be used instead.".format(x_max))
        else:
            x_max = bounds[1]

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.axes[0]

    # If samples are available, use them as bounds
    if samples is not None:
        ax.hist(samples, bins = int(np.sqrt(len(samples))), histtype = 'step', density = True, label = '$\mathrm{Samples}$', log = True)
        if bounds is None:
            x_min_l, x_max_l = ax.get_xlim()
            x_min = np.max((x_min, x_min_l))
            x_max = np.min((x_max, x_max_l))
    xlim = (x_min, x_max)
    x    = np.linspace(x_min, x_max, n_pts)
    dx   = x[1]-x[0]
    
    probs = np.array([d.pdf(x) for d in draws])
    
    percentiles = [50, 5, 16, 84, 95]
    p = {}
    for perc in percentiles:
        p[perc] = np.percentile(probs, perc, axis = 0)
    norm = p[50].sum()*dx
    for perc in percentiles:
        p[perc] = p[perc]/norm

    # Samples (if available)
    if samples is not None:
        ylim = ax.get_ylim()
    else:
        ax.set_yscale('log')
    
    # CR
    ax.fill_between(x, p[95], p[5], color = 'mediumturquoise', alpha = 0.25)
    ax.fill_between(x, p[84], p[16], color = 'darkturquoise', alpha = 0.25)
    # Selection function (if available)
    if selfunc is not None:
        if callable(selfunc):
            f_x = selfunc(x)
        else:
            f_x = selfunc
    # Injection (if available)
    if injected is not None:
        if callable(injected):
            p_x = injected(x)
        else:
            p_x = injected
        if injected_label is not None:
            injected_label = '$'+injected_label+'$'
        ax.plot(x, p_x, lw = 0.5, color = 'red', label = injected_label)
        if selfunc is not None:
            filtered_p_x = p_x*f_x
            ax.plot(x, filtered_p_x/np.sum(filtered_p_x*dx), lw = 0.5, color = 'k', label = '$\mathrm{Selection\ effects}$')
        
    # Median
    if true_value is not None:
        if true_value_label is not None:
            true_value_label = '$'+true_value_label+'$'
        ax.axvline(true_value, ls = '--', color = 'r', lw = 0.5, label = true_value_label)
    ax.plot(x, p[50], lw = 0.7, color = 'steelblue', label = '${0}$'.format(median_label))
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
    ax.legend(loc = 0)
    
    fig.align_labels()
    
    if save:
        if subfolder:
            plot_folder = Path(out_folder, 'density')
            if not plot_folder.exists():
                try:
                    plot_folder.mkdir()
                except FileExistsError:
                    # Avoids issue with parallelisation
                    pass
            log_folder = Path(out_folder, 'log_density')
            if not log_folder.exists():
                try:
                    log_folder.mkdir()
                except FileExistsError:
                    pass
            txt_folder = Path(out_folder, 'txt')
            if not txt_folder.exists():
                try:
                    txt_folder.mkdir()
                except FileExistsError:
                    pass
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
    if selfunc is not None:
        for perc in percentiles:
            p[perc] = np.percentile((probs/f_x).T, perc, axis = 1)
        norm = p[50].sum()*dx
        for perc in percentiles:
            p[perc] = p[perc]/norm
        
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        # CR
        ax.fill_between(x, p[95], p[5], color = 'mediumturquoise', alpha = 0.25)
        ax.fill_between(x, p[84], p[16], color = 'darkturquoise', alpha = 0.25)
        if injected is not None:
            # Injection
            ax.plot(x, p_x, lw = 0.5, color = 'red', label = injected_label)
        # Median
        ax.plot(x, p[50], lw = 0.7, color = 'steelblue', label = '${0}$'.format(median_label))
        if label is None:
            label = 'x'
        if unit is None or unit == '':
            ax.set_xlabel('${0}$'.format(label))
        else:
            ax.set_xlabel('${0}\ [{1}]$'.format(label, unit))
        ax.set_ylabel('$p({0})$'.format(label))
        ax.autoscale(True)
        ax.set_ylim(bottom = 1e-5, top = np.max(p[95])*1.1)
        ax.legend(loc = 0)
        if save:
            fig.savefig(Path(log_folder, 'log_true_{0}.pdf'.format(name)), bbox_inches = 'tight')
            ax.set_yscale('linear')
            ax.autoscale(True)
            fig.savefig(Path(plot_folder, 'true_{0}.pdf'.format(name)), bbox_inches = 'tight')
            np.savetxt(Path(txt_folder, 'prob_true_{0}.txt'.format(name)), np.array([x, p[50], p[5], p[16], p[84], p[95]]).T, header = 'x 50 5 16 84 95')
        if show:
            ax.set_yscale('linear')
            ax.autoscale(True)
            plt.show()
        plt.close()
    return fig

def plot_multidim(draws, samples = None, bounds = None, out_folder = '.', name = 'density', labels = None, units = None, hierarchical = False, show = False, save = True, subfolder = False, n_pts = 200, true_value = None, levels = [0.5, 0.68, 0.9], scatter_points = False, median_label = None, fig = None):
    """
    Plot the recovered multidimensional distribution along with samples from the true distribution (if available) as corner plot.
    
    Arguments:
        iterable draws:                container for mixture instances
        np.ndarray samples:            samples from the true distribution (if available)
        iterable bounds:               bounds for the recovered distribution. If None, bounds from mixture instances are used.
        str or Path out_folder:        output folder
        str name:                      name to be given to outputs
        list-of-str labels:            LaTeX-style quantity label, for plotting purposes
        list-of-str units:             LaTeX-style quantity unit, for plotting purposes
        bool hierarchical:             hierarchical inference, for plotting purposes
        bool save:                     whether to save the plot or not
        bool show:                     whether to show the plot during the run or not
        bool subfolder:                whether to save in a dedicated subfolder
        int n_pts:                     number of grid points (same for each dimension)
        iterable true_value:           true value to plot
        iterable levels:               credible levels to plot
        bool scatter_points:           scatter samples on 2d plots
        str median_label:              label to assign to the reconstruction
        matplotlib.figure.Figure fig:  figure to use for plotting. Must have (dim,dim) axes.
    
    Returns:
        matplotlib.figure.Figure: figure with the plot
    """
    
    dim = draws[0].dim
    if median_label is None:
        if hierarchical:
            median_label = '\mathrm{(H)DPGMM}'
        else:
            median_label = '\mathrm{DPGMM}'
    if labels is None:
        labels = ['$x_{'+'{0}'.format(i+1)+'}$' for i in range(dim)]
    else:
        labels = ['${0}$'.format(l) for l in labels]
    
    if units is not None:
        labels = [l[:-1]+'\ [{0}]$'.format(u) if not u == '' else l for l, u in zip(labels, units)]
    
    levels = np.atleast_1d(levels)
    
    ext_bounds = False
    if bounds is not None:
        ext_bounds = True
    all_bounds = np.atleast_2d([d.bounds for d in draws])
    x_min = np.min(all_bounds, axis = -1).max(axis = 0)
    x_max = np.max(all_bounds, axis = -1).min(axis = 0)
    
    probit = np.array([d.probit for d in draws]).any()
    
    if bounds is not None:
        bounds = np.atleast_2d(bounds)
        if bounds.shape == (1, 2):
            bounds = np.array([bounds[0] for _ in range(dim)])
        elif bounds.shape == (dim, 2):
            if not (bounds[:,0] >= x_min).all() and probit:
                warnings.warn("The provided lower bound is invalid for at least one draw. Default values will be used instead.")
            x_min[np.where(bounds[:,0] >= x_min)] = bounds[:,0][np.where(bounds[:,0] >= x_min)]
            if not (bounds[:,1] <= x_max).all() and probit:
                warnings.warn("The provided upper bound is invalid for at least one draw. Default values will be used instead.")
            x_max[np.where(bounds[:,1] <= x_max)] = bounds[:,1][np.where(bounds[:,1] <= x_max)]
        else:
            warnings.warn("The provided bounds have an invalid shape {0}. Shape must be (1,2) or (dim, 2).\nDefault bounds will be used instead.".format(bounds.shape))
    bounds = np.array([x_min, x_max]).T
    
    K = dim
    factor = 3.0          # size of one side of one panel
    lbdim = 0.5 * factor  # size of left/bottom margin
    trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.2         # w/hspace size
    plotdim = factor * dim + factor * (K - 1.0) * whspace
    dim_plt = lbdim + plotdim + trdim
    
    if fig is None:
        fig, axs = plt.subplots(K, K, figsize=(dim_plt, dim_plt))
    else:
        axs = np.array(fig.axes).reshape(dim,dim)
    # Format the figure.
    lb = lbdim / dim_plt
    tr = (lbdim + plotdim) / dim_plt
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace)
    # Samples (if available)
    if samples is not None:
        bins = np.array([int(np.sqrt(len(samples[:, c]))) for c in range(dim)])
        corner(samples, color = '#1f77b4', fig = fig, hist_kwargs = {'density': True, 'label':'$\mathrm{Samples}$', 'linewidth':0.7} , plot_density = False, contour_kwargs = {'linewidths':0.3, 'linestyles':'dashed'}, levels = [0.5,0.68,0.9], no_fill_contours = True, hist_bin_factor = bins/20, quiet = True)
        
    # 1D plots (diagonal)
    for column in range(K):
        ax = axs[column, column]
        # Marginalise over all uninterested columns
        dims = list(np.arange(dim))
        dims.remove(column)
        marg_draws = marginalise(draws, dims)
        # Credible regions
        lim = bounds[column]
        if samples is not None and not ext_bounds:
            lim_l = ax.get_xlim()
            lim[0] = np.max((lim[0], lim_l[0]))
            lim[1] = np.min((lim[1], lim_l[1]))
            
        x   = np.linspace(lim[0], lim[1], n_pts)
        dx  = x[1]-x[0]
        
        probs = np.array([d.pdf(x) for d in marg_draws])
        
        percentiles = [50, 5, 16, 84, 95]
        p = {}
        for perc in percentiles:
            p[perc] = np.percentile(probs, perc, axis = 0)
        norm = p[50].sum()*dx
        for perc in percentiles:
            p[perc] = p[perc]/norm
            
        # CR
        ax.fill_between(x, p[95], p[5], color = 'mediumturquoise', alpha = 0.25)
        ax.fill_between(x, p[84], p[16], color = 'darkturquoise', alpha = 0.25)
        if true_value is not None:
            if true_value[column] is not None:
                ax.axvline(true_value[column], c = 'orangered', lw = 0.5)
        ax.plot(x, p[50], lw = 0.7, color = 'steelblue', label = '${0}$'.format(median_label))
        ax.set_yticks([])
        if column == K - 1:
            if labels is not None:
                ax.set_xlabel(labels[-1])
        ticks = np.linspace(lim[0], lim[1], 5)
        ax.set_xticks(ticks)
        [l.set_rotation(45) for l in ax.get_xticklabels()]
        ax.set_xlim(lim[0], lim[1])
        if column < K-1:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    # 2D plots (off-diagonal)
    for row in range(K):
        for column in range(K):
            ax = axs[row,column]
            if column > row:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif column == row:
                continue
            
            # Marginalise
            dims = list(np.arange(dim))
            dims.remove(column)
            dims.remove(row)
            marg_draws = marginalise(draws, dims)
            
            # Credible regions
            lim = bounds[[row, column]]
            if samples is not None and not ext_bounds:
                lim_l = np.array([ax.get_ylim(), ax.get_xlim()])
                for i in range(2):
                    lim[i,0] = np.max((lim[i,0], lim_l[i,0]))
                    lim[i,1] = np.min((lim[i,1], lim_l[i,1]))
                
            grid, dgrid = recursive_grid(lim[::-1], np.ones(2, dtype = int)*int(n_pts))
            
            x = np.linspace(lim[0,0], lim[0,1], n_pts)
            y = np.linspace(lim[1,0], lim[1,1], n_pts)
            
            dd = np.array([d.pdf(grid) for d in marg_draws])
            median = np.percentile(dd, 50, axis = 0)
            median = median/(median.sum()*np.prod(dgrid))
            median = median.reshape(n_pts, n_pts)
            
            X,Y = np.meshgrid(x,y)
            with np.errstate(divide = 'ignore'):
                logmedian = np.nan_to_num(np.log(median), nan = -np.inf, neginf = -np.inf)
            _,_,levs = ConfidenceArea(logmedian, x, y, adLevels=levels)
            ax.contourf(Y, X, np.exp(logmedian), cmap = 'Blues', levels = 100)
            if true_value is not None:
                if true_value[row] is not None:
                    ax.axhline(true_value[row], c = 'orangered', lw = 0.5)
                if true_value[column] is not None:
                    ax.axvline(true_value[column], c = 'orangered', lw = 0.5)
                if true_value[column] is not None and true_value[row] is not None:
                    ax.plot(true_value[column], true_value[row], color = 'orangered', marker = 's', ms = 3)
            c1 = ax.contour(Y, X, logmedian, np.sort(levs), colors='steelblue', linewidths=0.3)
            if plot_settings.tex_flag:
                ax.clabel(c1, fmt = {l:'{0:.0f}\\%'.format(100*s) for l,s in zip(c1.levels, np.sort(levels)[::-1])}, fontsize = 3)
            else:
                ax.clabel(c1, fmt = {l:'{0:.0f}\%'.format(100*s) for l,s in zip(c1.levels, np.sort(levels)[::-1])}, fontsize = 3)
            # Samples (if available)
            if samples is not None and scatter_points:
                ax.scatter(samples[:,column], samples[:,row], marker = '+', c = 'orangered', linewidths = 1)
            # Ticks
            xticks = np.linspace(lim[1,0], lim[1,1], 5)
            ax.set_xticks(xticks)
            yticks = np.linspace(lim[0,0], lim[0,1], 5)
            ax.set_yticks(yticks)
            ax.set_xlim(*lim[1])
            ax.set_ylim(*lim[0])
            if column == 0:
                ax.set_ylabel(labels[row])
                [l.set_rotation(45) for l in ax.get_yticklabels()]
            else:
                ax.set_yticklabels([])
            if row == K - 1:
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                ax.set_xlabel(labels[column])
            else:
                ax.set_xticklabels([])
    fig.axes[K-1].legend(*fig.axes[0].get_legend_handles_labels(), loc = 'center')
    fig.align_labels()
    
    if show:
        plt.show()
    if save:
        if not subfolder:
            fig.savefig(Path(out_folder, '{0}.pdf'.format(name)), bbox_inches = 'tight')
        else:
            if not Path(out_folder, 'density').exists():
                try:
                    Path(out_folder, 'density').mkdir()
                except FileExistsError:
                    pass
            fig.savefig(Path(out_folder, 'density', '{0}.pdf'.format(name)), bbox_inches = 'tight')
    plt.close()
    return fig
    
def plot_1d_dist(x, draws, injected = None, samples = None, out_folder = '.', name = 'density', label = None, unit = None, show = False, save = True, subfolder = False, true_value = None, true_value_label = '\mathrm{True\ value}', injected_label = '\mathrm{Simulated}', median_label = '\mathrm{Median}', logx = False, logy = False):
    """
    Plot a 1D distribution along with samples from the true distribution (if available).
    Differently from plot_median_cr, this method requires the distribution to be already evaluated.
    For FIGARO mixture instances, please use plot_median_cr.

    Arguments:
        iterable x:                      values at which realisations are evaluated
        iterable draws:                  container for realisations
        callable or np.ndarray injected: injected distribution (if available)
        np.ndarray samples:              samples from the true distribution (if available)
        str or Path out_folder:          output folder
        str name:                        name to be given to outputs
        str label:                       LaTeX-style quantity label, for plotting purposes
        str unit:                        LaTeX-style quantity unit, for plotting purposes
        bool save:                       whether to save the plots or not
        bool show:                       whether to show the plots during the run or not
        bool subfolder:                  whether to save the plots in different subfolders (for multiple events)
        float true_value:                true value to infer
        str true_value_label:            label to assign to the true value marker
        str injected_label:              label to assign to the injected distribution
        str median_label:                label to assign to the median distribution
        bool logx:                       x log scale
        bool logy:                       y log scale
    """
    
    if not np.shape(x)[0] == np.shape(draws)[-1]:
        raise ValueError("x and each draw must have the same length")
    
    percentiles = [50, 5, 16, 84, 95]
    p = {}
    for perc in percentiles:
        p[perc] = np.percentile(draws, perc, axis = 0)
    
    fig, ax = plt.subplots()
    
    # Samples (if available)
    if samples is not None:
        ax.hist(samples, bins = int(np.sqrt(len(samples))), histtype = 'step', density = True, label = '$\mathrm{Samples}$')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
    
    # CR
    ax.fill_between(x, p[95], p[5], color = 'mediumturquoise', alpha = 0.25)
    ax.fill_between(x, p[84], p[16], color = 'darkturquoise', alpha = 0.25)
    # Injection (if available)
    if injected is not None:
        if callable(injected):
            p_x = injected(x)
        else:
            p_x = injected
        if injected_label is not None:
            injected_label = '$'+injected_label+'$'
        ax.plot(x, p_x, lw = 0.5, color = 'red', label = injected_label)
        
    # Median
    if true_value is not None:
        if true_value_label is not None:
            true_value_label = '$'+true_value_label+'$'
        ax.axvline(true_value, ls = '--', color = 'r', lw = 0.5, label = true_value_label)
    if median_label is not None:
        median_label = '$'+median_label+'$'
    ax.plot(x, p[50], lw = 0.7, color = 'steelblue', label = median_label)
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
    ax.legend(loc = 0)
    if logy:
        ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')
    
    fig.align_labels()
    
    if save:
        if subfolder:
            plot_folder = Path(out_folder, 'density')
            if not plot_folder.exists():
                try:
                    plot_folder.mkdir()
                except FileExistsError:
                    # Avoids issue with parallelisation
                    pass
            txt_folder = Path(out_folder, 'txt')
            if not txt_folder.exists():
                try:
                    txt_folder.mkdir()
                except FileExistsError:
                    pass
        else:
            plot_folder = out_folder
            txt_folder  = out_folder
        ax.autoscale(True)
        if samples is not None:
            ax.set_xlim(xlim)
        fig.savefig(Path(plot_folder, '{0}.pdf'.format(name)), bbox_inches = 'tight')
        np.savetxt(Path(txt_folder, 'prob_{0}.txt'.format(name)), np.array([x, p[50], p[5], p[16], p[84], p[95]]).T, header = 'x 50 5 16 84 95')
    if show:
        ax.autoscale(True)
        if samples is not None:
            ax.set_xlim(xlim)
        plt.show()
    plt.close()
    

def plot_n_clusters_alpha(n_cl, alpha, out_folder = '.', name = 'event', show = False, save = True):
    """
    Plot the number of clusters and the concentration parameter as functions of the number of samples.
    
    Arguments:
        np.ndarray n_cl:        number of active clusters
        np.ndarray alpha:       concentration parameter
        str or Path out_folder: output folder
        str name:               name to be given to outputs
        bool save:              whether to save the plot or not
        bool show:              whether to show the plot during the run or not
    """
    fig, ax = plt.subplots()
    ax1 = ax.twinx()
    ax.plot(np.arange(1, len(n_cl)+1), n_cl, ls = '--', marker = '', lw = 0.7, color = 'k')
    ax1.plot(np.arange(1, len(alpha)+1), alpha, ls = '--', marker = '', lw = 0.7, color = 'r')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$N_{\mathrm{cl}}(t)$', color = 'k')
    ax1.set_ylabel('$\alpha(t)$', color = 'r')
    if show:
        plt.show()
    if save:
        fig.savefig(Path(out_folder, '{0}_n_cl_alpha.pdf'.format(name)), bbox_inches = 'tight')
    plt.close()

def pp_plot_cdf(draws, injection, n_points = 1000, out_folder = '.', name = 'event', show = False, save = True):
    """
    Make pp-plot comparing draws cdfs and injection cdf
    
    Arguments:
        iterable draws:         container of mixture instances
        callable injection:     injected density
        int n_points:           number of points for linspace
        str or Path out_folder: output folder
        str name:               name to be given to outputs
        bool save:              whether to save the plot or not
        bool show:              whether to show the plot during the run or not
    """
    all_bounds = np.atleast_2d([d.bounds[0] for d in draws])
    x_min = np.max(all_bounds[:,0])
    x_max = np.min(all_bounds[:,1])
    x = np.linspace(x_min, x_max, n_points)
    
    functions     = np.array([mix(x) for mix in draws])
    median        = np.percentile(functions, 50, axis = 0)
    cdf_draws     = np.array([fast_cumulative(d) for d in functions])
    cdf_median    = fast_cumulative(median)
    cdf_injection = fast_cumulative(injection(x))
    
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection = 'pp_plot')
    ax.add_confidence_band(len(cdf_median), cl=0.9, color = 'ghostwhite')
    ax.add_diagonal()
    for cdf in cdf_draws:
        ax.plot(cdf_injection, cdf, lw = 0.5, alpha = 0.5, color = 'darkturquoise')
    ax.plot(cdf_injection, cdf, color = 'steelblue', lw = 0.7)
    ax.set_xlabel('$\mathrm{Injected}$')
    ax.set_ylabel('$\mathrm{FIGARO}$')
    if show:
        plt.show()
    if save:
        fig.savefig(Path(out_folder, '{0}_ppplot.pdf'.format(name)), bbox_inches = 'tight')
    plt.close()

def pp_plot_levels(CR_levels, median_CR = None, out_folder = '.', name = 'MDC', show = False, save = True):
    """
    Make pp-plot.
    
    Arguments:
        iterable CR:            2D array with credible levels for each event
        iterable median_CR:     credible levels of medians
        str or Path out_folder: output folder
        str name:               name to be given to outputs
        bool save:              whether to save the plot or not
        bool show:              whether to show the plot during the run or not
    """
    CR_levels = np.atleast_1d(CR_levels)
    if len(CR_levels.shape) > 1:
        CR_levels = CR_levels.T
    n_evs     = CR_levels.shape[-1]
    L         = np.linspace(0,1,n_evs+2)
    
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection = 'pp_plot')
    ax.add_confidence_band(n_evs, zorder = n_evs)
    ax.add_diagonal(zorder = n_evs+1)
    if len(CR_levels.shape) > 1:
        sorted = []
        for cr in CR_levels:
            if median_CR is not None:
                lw = 0.3
                c  = 'lightsteelblue'
            else:
                lw = 0.6
                c  = 'steelblue'
            x = np.append(0, np.append(cr, 1))
            ax.plot(np.sort(x), L, lw = lw, alpha = 0.5, color = c)
        if median_CR is not None:
            x = np.append(0, np.append(median_CR, 1))
            ax.plot(np.sort(x), L, lw = 0.8, color = 'steelblue', label = '$\mathrm{Median}$', zorder = n_evs+2)
        # Add label for draws
        handles, labels = ax.get_legend_handles_labels()
        line = Line2D([0], [0], label='$\mathrm{Draws}$', lw = lw, color = c)
        handles.extend([line])
        ax.legend(handles=handles, loc = 0, frameon = False)
    else:
        x = np.append(0, np.append(CR_levels, 1))
        ax.plot(np.sort(x), L, lw = 0.8, color = 'steelblue', zorder = n_evs+2)
    # Maquillage
    ax.set_xlabel('$P$')
    ax.set_ylabel('$\mathrm{Fraction\ of\ events\ within\ }CR_P$')
    if show:
        plt.show()
    if save:
        fig.savefig(Path(out_folder, '{0}_ppplot.pdf'.format(name)), bbox_inches = 'tight')
    plt.close()

def joyplot(draws, x_values, y_values, credible_regions = False, fill = True, solid = False, overlap = 1., xlabel = None, ylabel = None, xunit = None, yunit = None, colormap = 'coolwarm', out_folder = '.', name = 'joyplot', subfolder = False, show = False, save = True, joy = False):
    """
    Make a joyplot (also known as ridgeline plot) of a set of distributions.
    Heavily inspired by leotac's JoyPy (https://github.com/leotac/joypy).
    
    Arguments:
        np.ndarray draws:       (n_y, n_x) or (n_y, n_draws, n_x) array containing the distributions to plot
        np.ndarray x_values:    x values at which the distributions are evaluated
        np.ndarray y_values:    y values corresponding to the different distributions
        bool credible_regions:  whether to plot credible regions or not
        bool fill:              whether to fill the space below the distribution or not (ignored if credible_regions is True)
        bool solid:             wheter to fill with transparent or solid colour
        float overlap:          space between different plots
        str xlabel:             x axis label (LaTeX style)
        str ylabel:             colorbar label (LaTeX style)
        str xunit:              x axis units (LaTeX style) - default: no units
        str yunit:              colorbar units (LaTeX style) - default: no units
        str colormap:           a valid matplotlib colormap name
        str or Path out_folder: output folder
        str name:               name to be given to outputs
        bool subfolder:         whether to save the plots in different subfolders (for multiple events)
        bool save:              whether to save the plots or not
        bool show:              whether to show the plots during the run or not
        bool joy:               format the plot to look like the cover of Joy Division's "Unknown Pleasures"
        
    Returns:
        matplotlib.figure.Figure: figure with the plot
    """
    # Labels
    if xlabel is None:
        xlabel = '$x$'
    else:
        xlabel = '${0}$'.format(xlabel)
    if xunit is not None:
        xlabel = xlabel[:-1]+'\ [{0}]$'.format(xunit)
    if ylabel is None:
        ylabel = '$y$'
    else:
        ylabel = '${0}$'.format(ylabel)
    if yunit is not None:
        ylabel = ylabel[:-1]+'\ [{0}]$'.format(yunit)
    
    # Check that we have as many y values as different draws evaluations
    if not len(y_values) == np.shape(draws)[0]:
        raise ValueError("y_values and draws must have the same length")

    color = iter(colormaps[colormap](np.linspace(1, 0, len(y_values))))
    cmappable = plt.cm.ScalarMappable(norm=Normalize(np.min(y_values),np.max(y_values)), cmap=colormap)
    num_axes = len(y_values)

    # Each plot will have its own axis
    fig = plt.figure()
    if joy:
        fig.patch.set_facecolor('k')
    gs = GridSpec(len(y_values), 17, figure = fig)
    _axes = []
    [_axes.append(fig.add_subplot(gs[i, :-1])) for i in range(len(y_values))]
    if not joy:
        _axes.append(fig.add_subplot(gs[:, -1]))
        # Global colorbar
        cbar    = fig.colorbar(cmappable, cax = _axes[-1])
        cbar.set_label(ylabel)
    if joy:
        [ax.set_facecolor('k') for ax in _axes]
    for i, dd in enumerate(draws[::-1]):
        ax = _axes[i]
        if not joy:
            c = next(color)
        else:
            c = 'w'
        percentiles = [50, 16, 84]
        p = {}
        if len(np.shape(draws)) == 3:
            for perc in percentiles:
                p[perc] = np.percentile(dd, perc, axis = 0)
            if credible_regions and not joy:
                ax.fill_between(x_values, p[84], p[16], color = c, alpha = 0.25)
        else:
            p[50] = dd
        if fill and not credible_regions:
            if solid:
                alpha = 1
            else:
                alpha = 0.25
            if not joy:
                ax.fill_between(x_values, p[50], np.zeros(len(x_values)), color = c, alpha = alpha)
            else:
                ax.fill_between(x_values, p[50], np.zeros(len(x_values)), color = 'k', alpha = 1)
        ax.plot(x_values, p[50], clip_on = True, c = c)

        # Setup the current axis: transparency, labels, spines.
        ax.yaxis.grid(False)
        ax.patch.set_alpha(0)
        ax.tick_params(axis='both', which='both', length=0, pad=10)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)

    # Final adjustments
    # Compute a final axis, used to apply global settings
    last_axis = fig.add_subplot(gs[:, :-1])
    if joy:
        last_axis.set_facecolor('k')

    for side in ['top', 'bottom', 'left', 'right']:
        last_axis.spines[side].set_visible(False)

    # This looks hacky, but all the axes share the x-axis,
    # so they have the same lims and ticks
    last_axis.set_xlim(_axes[0].get_xlim())
    last_axis.set_xticks(np.array(_axes[0].get_xticks()[1:-1]))
    for t in last_axis.get_xticklabels():
        if not joy:
            t.set_visible(True)
        else:
            t.set_visible(False)
    else:
        if not joy:
            last_axis.xaxis.set_visible(True)
        else:
            last_axis.xaxis.set_visible(False)
    last_axis.yaxis.set_visible(False)
    last_axis.grid(False)

    # Last axis on the back
    last_axis.zorder = min(a.zorder for a in _axes) - 1
    _axes = list(_axes) + [last_axis]
    last_axis.set_xlabel(xlabel)

    # The magic overlap happens here.
    h_pad = 5 + (-5*(1+overlap))
    fig.tight_layout(h_pad=h_pad)

    if show:
        plt.show()
    if save:
        if not subfolder:
            fig.savefig(Path(out_folder, '{0}.pdf'.format(name)), bbox_inches = 'tight')
        else:
            if not Path(out_folder, 'density').exists():
                try:
                    Path(out_folder, 'density').mkdir()
                except FileExistsError:
                    pass
            fig.savefig(Path(out_folder, 'density', '{0}.pdf'.format(name)), bbox_inches = 'tight')
    plt.close()
    return fig
