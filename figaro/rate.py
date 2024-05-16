import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from collections import Counter
from scipy.stats import poisson, norm
from figaro.cosmology import dVdz_approx_planck18

def sample_rate(draws, n_obs, selfunc, T, volume = None, size = None, each = False, n_draws = 1e4, dvdz = None, z_index = -1, normalise_alpha = False):
    """
    Compute the integrated rate given a set of draws. If volume keyword is not used, returns the expected number of events before filtering.
    
    Arguments:
        iterable draws:       container for mixture instances
        int n_obs:            number of observed events
        callable selfunc:     selection function approximant
        double T:             duration of the observations
        double volume:        surveyed volume. If None, it is estimated.
        int size:             number of samples
        bool each:            sample one point from each draw
        int n_draws:          number of MC draws
        callable dvdz         function to compute the comoving volume element (in Gpc^3)
        int z_index           redshift parameter index (default: last dimension)
        bool normalise_alpha: re-normalise alpha (required if the selection function includes dV/dz*(1+z)^-1)
    
    Returns:
        np.ndarray: integrated rate samples
    """
    if size is None or each:
        size = len(draws)
    if volume is None:
        volume = sample_VT(draws, selfunc, T, size = size, each = each, dvdz = dvdz, z_index = z_index)
    if normalise_alpha:
        normalise_alpha_factor(draws, dvdz = dvdz, z_index = z_index, n_draws = n_draws)
    N_exp   = [poisson(int(n_obs/d.alpha_factor)) for d in draws]
    if each:
        return np.array([N.mean()/v for N, v in zip(N_exp, volume)])
    idx     = np.random.choice(np.arange(len(N_exp)), size = int(size))
    ctr     = Counter(idx)
    samples = np.empty(shape = (1,))
    for i, n in zip(ctr.keys(), ctr.values()):
        samples = np.concatenate((samples, N_exp[i].rvs(n)))
    return np.divide(samples[1:], volume)

def normalise_alpha_factor(draws, dvdz = None, z_index = -1, z_max = None, n_draws = 1e4):
    """
    Normalise the alpha factor for each DPGMM realisation. This is required if during the reconstruction the selection function included the dV/dz*(1+z)^-1 term.
    
    Arguments:
        iterable draws: container for mixture instances
        callable dvdz:  function to compute the comoving volume element (in Gpc^3)
        int z_index:    redshift parameter index (default: last dimension)
        double z_max:   max redshift value
        int n_draws:    number of MC draws
    """
    if dvdz is None:
        dvdz = dVdz_approx_planck18
    if z_max is not None:
        reg_const = (1+z_max)/dvdz(z_max)
    else:
        reg_const = 1.
    for d in draws:
        ss = d.rvs(int(n_draws))
        d.alpha_factor /= np.mean(reg_const*dvdz(ss[:,z_index])/(1+ss[:,z_index]))

def sample_VT(draws, selfunc, T, size = None, n_draws = 1e4, each = False, dvdz = None, z_index = -1):
    """
    Draw samples from the VT distribution given the astrophysical distribution of quantities, the selection function and the observing time.
    See Eqs. (21) of Kapadia et al (2020) (https://arxiv.org/pdf/1903.06881.pdf).
    The calibration uncertainty is estimated as in Sec. 5 - Eq. (17) - of Abbott et al (2016) (https://arxiv.org/pdf/1606.03939.pdf)
    
    Arguments:
        iterable draws:   container for mixture instances
        callable selfunc: selection function approximant
        double T:         duration of the observations
        int size:         number of samples
        int n_draws:      number of MC draws
        bool each:        sample one point from each draw
        callable dvdz     function to compute the comoving volume element (in Gpc^3)
        int z_index       redshift parameter index (default: last dimension)
    
    Returns:
        np.ndarray: VT samples
    """
    draws = np.atleast_1d(draws)
    if size is None or each:
        size = len(draws)
    if dvdz is None:
        dvdz = dVdz_approx_planck18
    VT_dist = []
    for d in draws:
        ss     = d.rvs(int(n_draws))
        comvol = dvdz(ss[:,z_index])
        sf     = selfunc(ss)
        VT_s   = T*sf*comvol/(1+ss[:,z_index])
        VT     = np.mean(VT_s)
        S      = np.sqrt(np.cov(VT_s)/n_draws)
        VT_dist.append(norm(np.log(VT),S/VT))
    if each:
        return np.array([np.exp(VT_i.mean()) for VT_i in VT_dist])
    else:
        idx     = np.random.choice(np.arange(len(draws)), size = int(size))
        ctr     = Counter(idx)
        samples = np.empty(shape = (1,))
        for i, n in zip(ctr.keys(), ctr.values()):
            samples = np.concatenate((samples, np.exp(VT_dist[i].rvs(n))))
        return samples[1:]

def plot_integrated_rate(samples, out_folder = '.', name = 'density', volume_unit = '\\mathrm{Gpc}^{-3}\\mathrm{yr}^{-1}', show = False, save = True, true_value = None, true_label = None):
    """
    Plot the integrated rate distribution.
    
    Arguments:
        np.ndarray samples:     rate samples
        str or Path out_folder: output folder
        str name:               name to be given to output
        str volume_unit:        LaTeX-style volume unit, for plotting purposes
        bool save:              whether to save the plot or not
        bool show:              whether to show the plot during the run or not
        iterable true_value:    true value to plot
        iterable levels:        credible levels to plot
        str true_label:         label to assign to the reconstruction
    
    Returns:
        matplotlib.figure.Figure: figure with the plot
    """
    if true_label is None:
        true_label = '$\\mathcal{R}_\\mathrm{true}$'
    else:
        true_label = f'${true_label}$'
    if volume_unit is not None:
        xlabel = '$\\mathcal{R}\ ['+volume_unit+']$'
    else:
        xlabel = '$\\mathcal{R}$'
    fig, ax = plt.subplots()
    if true_value is not None:
        ax.axvline(true_value, color = 'firebrick', dashes = (5,5), label = true_label)
    ax.hist(samples, histtype = 'step', density = True)
    if true_value is not None:
        ax.legend(loc = 0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$p(\\mathcal{R})$')
    if save:
        fig.savefig(Path(out_folder, 'integrated_rate_{0}.pdf'.format(name)), bbox_inches = 'tight')
    if show:
        plt.show()
    plt.close()
    return fig

def plot_differential_rate(draws, rate, injected = None, bounds = None, out_folder = '.', name = 'density', n_pts = 1000, label = None, unit = None, volume_unit = '\\mathrm{Gpc}^{-3}\\mathrm{yr}^{-1}', hierarchical = False, show = False, save = True, true_value = None, true_value_label = '\mathrm{True\ value}', injected_label = '\mathrm{Simulated}', median_label = None, fig = None, colors = ['steelblue', 'darkturquoise', 'mediumturquoise']):
    """
    Plot the recovered 1D distribution along with the injected distribution and samples from the true distribution (both if available).
    
    Arguments:
        iterable draws:                  container for mixture instances
        np.ndarray rate:                 rate samples or value
        callable or np.ndarray injected: injected distribution (if available)
        iterable bounds:                 bounds for the recovered distribution. If None, bounds from mixture instances are used.
        str or Path out_folder:          output folder
        str name:                        name to be given to outputs
        int n_pts:                       number of points for linspace
        str label:                       LaTeX-style quantity label, for plotting purposes
        str unit:                        LaTeX-style quantity unit, for plotting purposes
        str volume_unit:                 LaTeX-style volume unit, for plotting purposes
        bool hierarchical:               hierarchical inference, for plotting purposes
        bool save:                       whether to save the plots or not
        bool show:                       whether to show the plots during the run or not
        float true_value:                true value to infer
        str true_value_label:            label to assign to the true value marker
        str injected_label:              label to assign to the injected distribution
        str median_label:                label to assign to the reconstruction
        matplotlib.figure.Figure fig:    figure to use for plotting. Must have (dim,dim) axes.
        list-of-str colors:              list of colors for median, 68% and 90% credible regions
    
    Returns:
        matplotlib.figure.Figure: figure with the plot
    """
    if median_label is None:
        if hierarchical:
            median_label = '\mathrm{(H)DPGMM}'
        else:
            median_label = '\mathrm{DPGMM}'
    
    if not hasattr(rate, '__iter__'):
        rate = np.ones(len(draws))*rate
    
    color_med, color_68, color_90 = colors
    
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

    xlim  = (x_min, x_max)
    x     = np.linspace(x_min, x_max, n_pts)
    dx    = x[1]-x[0]
    probs = np.array([r*d.pdf(x) for d, r in zip(draws, rate)])
    percentiles = [50, 5, 16, 84, 95]
    p = {}
    for perc in percentiles:
        p[perc] = np.percentile(probs, perc, axis = 0)
    
    ax.set_yscale('log')
    # CR
    ax.fill_between(x, p[95], p[5], color = color_90, alpha = 0.25)
    ax.fill_between(x, p[84], p[16], color = color_68, alpha = 0.25)
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
        ax.axvline(true_value, ls = '--', color = 'firebrick', dashes = (5,5), lw = 0.5, label = true_value_label)
    ax.plot(x, p[50], lw = 0.7, color = color_med, label = '${0}$'.format(median_label))
    if label is None:
        label = 'x'
    if unit is None or unit == '':
        ax.set_xlabel('${0}$'.format(label))
        unit_label = volume_unit
    else:
        ax.set_xlabel('${0}\ [{1}]$'.format(label, unit))
        unit_label = volume_unit +unit+'^{-1}'
    y_label = '$\\frac{\\mathrm{d}\\mathcal{R}}{\\mathrm{d}'+label+'}\ ['+unit_label+']$'
    ax.set_ylabel(y_label)
    ax.set_ylim(bottom = 1e-5, top = np.max(p[95])*1.1)
    ax.legend(loc = 0)
    
    fig.align_labels()
    
    if save:
        plot_folder = out_folder
        log_folder  = out_folder
        txt_folder  = out_folder
        fig.savefig(Path(log_folder, 'log_differential_rate_{0}.pdf'.format(name)), bbox_inches = 'tight')
        ax.set_yscale('linear')
        ax.autoscale(True)
        fig.savefig(Path(plot_folder, 'differential_rate_{0}.pdf'.format(name)), bbox_inches = 'tight')
        np.savetxt(Path(txt_folder, 'prob_differential_rate_{0}.txt'.format(name)), np.array([x, p[50], p[5], p[16], p[84], p[95]]).T, header = 'x 50 5 16 84 95')
    if show:
        ax.set_yscale('linear')
        ax.autoscale(True)
        plt.show()
    plt.close()
    return fig

