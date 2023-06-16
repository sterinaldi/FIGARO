import numpy as np
import warnings
import importlib

import optparse as op
import dill

from pathlib import Path
from tqdm import tqdm

from figaro.mixture import DPGMM, HDPGMM
from figaro.utils import save_options, get_priors
from figaro.plot import plot_median_cr, plot_multidim, plot_1d_dist
from figaro.load import load_single_event, load_data, save_density, load_density
from figaro.diagnostic import compute_entropy_single_draw, compute_angular_coefficients
from figaro.exceptions import FIGAROException

def main():

    parser = op.OptionParser()
    # Input/output
    parser.add_option("-i", "--input", type = "string", dest = "samples_file", help = "File with samples")
    parser.add_option("-b", "--bounds", type = "string", dest = "bounds", help = "Density bounds. Must be a string formatted as '[[xmin, xmax], [ymin, ymax],...]'. For 1D distributions use '[xmin, xmax]'. Quotation marks are required and scientific notation is accepted", default = None)
    parser.add_option("-o", "--output", type = "string", dest = "output", help = "Output folder. Default: same directory as samples", default = None)
    parser.add_option("-j", dest = "json", action = 'store_true', help = "Save mixtures in json file", default = False)
    parser.add_option("--inj_density", type = "string", dest = "inj_density_file", help = "Python module with injected density - please name the method 'density'", default = None)
    parser.add_option("--parameter", type = "string", dest = "par", help = "GW parameter(s) to be read from file", default = None)
    parser.add_option("--waveform", type = "string", dest = "wf", help = "Waveform to load from samples file. To be used in combination with --parameter. Accepted values: 'combined', 'imr', 'seob'", default = 'combined')
    # Plot
    parser.add_option("--name", type = "string", dest = "h_name", help = "Name to be given to hierarchical inference files. Default: same name as samples folder parent directory", default = None)
    parser.add_option("-p", "--postprocess", dest = "postprocess", action = 'store_true', help = "Postprocessing", default = False)
    parser.add_option("--symbol", type = "string", dest = "symbol", help = "LaTeX-style quantity symbol, for plotting purposes", default = None)
    parser.add_option("--unit", type = "string", dest = "unit", help = "LaTeX-style quantity unit, for plotting purposes", default = None)
    parser.add_option("-n", "--no_plot_dist", dest = "plot_dist", action = 'store_false', help = "Skip distribution plot", default = True)
    parser.add_option("-s", "--save_se", dest = "save_single_event", action = 'store_true', help = "Save single event plots", default = False)
    parser.add_option("--hier_samples", type = "string", dest = "true_vals", help = "Samples from hierarchical distribution (true single-event values, for simulations only)", default = None)
    # Settings
    parser.add_option("--draws", type = "int", dest = "n_draws", help = "Number of draws", default = 100)
    parser.add_option("--se_draws", type = "int", dest = "n_se_draws", help = "Number of draws for single-event distribution. Default: same as hierarchical distribution", default = None)
    parser.add_option("--n_samples_dsp", type = "int", dest = "n_samples_dsp", help = "Number of samples to analyse (downsampling). Default: all", default = -1)
    parser.add_option("--exclude_points", dest = "exclude_points", action = 'store_true', help = "Exclude points outside bounds from analysis", default = False)
    parser.add_option("--cosmology", type = "string", dest = "cosmology", help = "Cosmological parameters (h, om, ol). Default values from Planck (2021)", default = '0.674,0.315,0.685')
    parser.add_option("--sigma_prior", dest = "sigma_prior", type = "string", help = "Expected standard deviation (prior) - single value or n-dim values. If None, it is estimated from samples", default = None)
    parser.add_option("--fraction", dest = "scale", type = "float", help = "Fraction of samples standard deviation for sigma prior. Overrided by sigma_prior.", default = None)
    parser.add_option("-e", "--events", dest = "run_events", action = 'store_false', help = "Skip single-event analysis", default = True)
    parser.add_option("--snr_threshold", dest = "snr_threshold", type = "float", help = "SNR threshold for simulated GW datasets", default = None)
    parser.add_option("--far_threshold", dest = "far_threshold", type = "float", help = "FAR threshold for simulated GW datasets", default = None)
    parser.add_option("--zero_crossings", dest = "zero_crossings", type = "int", help = "Number of zero-crossings of the entropy derivative to call the number of samples sufficient. Default as in Appendix B of Rinaldi & Del Pozzo (2021)", default = 5)
    parser.add_option("--window", dest = "window", type = "int", help = "Number of points to use to approximate the entropy derivative", default = 200)
    parser.add_option("--entropy_interval", dest = "entropy_interval", type = "int", help = "Number of samples between two entropy evaluations", default = 100)
    parser.add_option("--entropy_draws", dest = "entropy_draws", type = "string", help = "Number of monte carlo samples for entropy evaluation", default = '1e3')
    parser.add_option("--no_probit", dest = "probit", action = 'store_false', help = "Disable probit transformation", default = True)
    
    (options, args) = parser.parse_args()

    # Paths
    options.samples_file = Path(options.samples_file).resolve()
    if options.output is not None:
        options.output = Path(options.output).resolve()
        if not options.output.exists():
            options.output.mkdir(parents=True)
    else:
        options.output = options.samples_file.parent
    # Read bounds
    if options.bounds is not None:
        options.bounds = np.array(np.atleast_2d(eval(options.bounds)), dtype = np.float64)
    elif options.bounds is None and not options.postprocess:
        raise Exception("Please provide bounds for the inference (use -b '[[xmin,xmax],[ymin,ymax],...]')")
    # If provided, load injected density
    inj_density = None
    if options.inj_density_file is not None:
        inj_file_name = Path(options.inj_density_file).parts[-1].split('.')[0]
        spec = importlib.util.spec_from_file_location(inj_file_name, options.inj_density_file)
        inj_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inj_module)
        inj_density = inj_module.density
    # Read cosmology
    options.h, options.om, options.ol = (float(x) for x in options.cosmology.split(','))
    # Read parameter(s)
    if options.par is not None:
        options.par = options.par.split(',')
    # Read number of samples
    options.entropy_draws = int(eval(options.entropy_draws))
    # File extension
    if options.json:
        options.ext = 'json'
    else:
        options.ext = 'pkl'
        
    save_options(options, options.output)
    
    if options.samples_file.is_file():
        hier_flag = False
        # Load samples
        samples, name = load_single_event(options.samples_file, par = options.par, n_samples = options.n_samples_dsp, h = options.h, om = options.om, ol = options.ol, waveform = options.wf, snr_threshold = options.snr_threshold, far_threshold = options.far_threshold)
        n_pts = len(samples)
        try:
            dim = np.shape(samples)[-1]
        except IndexError:
            dim = 1
        if options.exclude_points:
            print("Ignoring points outside bounds.")
            samples = samples[np.where((np.prod(options.bounds[:,0] < samples, axis = 1) & np.prod(samples < options.bounds[:,1], axis = 1)))]
        else:
            # Check if all samples are within bounds
            if not np.alltrue([(samples[:,i] > options.bounds[i,0]).all() and (samples[:,i] < options.bounds[i,1]).all() for i in range(dim)]):
                raise ValueError("One or more samples are outside the given bounds.")
    else:
        hier_flag = True
        # Load events
        events, names = load_data(options.samples_file, par = options.par, n_samples = options.n_samples_dsp, h = options.h, om = options.om, ol = options.ol, waveform = options.wf, snr_threshold = options.snr_threshold)
        n_pts = len(events)
        try:
            dim = np.shape(events[0][0])[-1]
        except IndexError:
            dim = 1
        # Read hierarchical name
        if options.h_name is None:
            name = options.samples_folder.parent.parts[-1]
        else:
            name = options.h_name
        if options.exclude_points:
            print("Ignoring points outside bounds.")
            for i, ev in enumerate(events):
                events[i] = ev[np.where((np.prod(options.bounds[:,0] < ev, axis = 1) & np.prod(ev < options.bounds[:,1], axis = 1)))]
            all_samples = np.atleast_2d(np.concatenate(events))
        else:
            # Check if all samples are within bounds
            all_samples = np.atleast_2d(np.concatenate(events))
            if not np.alltrue([(all_samples[:,i] > options.bounds[i,0]).all() and (all_samples[:,i] < options.bounds[i,1]).all() for i in range(dim)]):
                raise ValueError("One or more samples are outside the given bounds.")
        output_plots = Path(options.output, 'plots')
        if not output_plots.exists():
            output_plots.mkdir()
        output_draws = Path(options.output, 'draws')
        if not output_draws.exists():
            output_draws.mkdir()
    # If provided, load true values
    true_vals = None
    if not hier_flag:
        true_vals = samples
    if hier_flag and options.true_vals is not None:
        options.true_vals = Path(options.true_vals).resolve()
        true_vals = np.loadtxt(options.true_vals)
    if options.sigma_prior is not None:
        options.sigma_prior = np.array([float(s) for s in options.sigma_prior.split(',')])
    # Entropy derivative
    if n_pts < options.window:
        options.window = n_pts//5
        warnings.warn("The window is smaller than the minimum recommended window for entropy derivative estimate. Results might be unreliable")
    if n_pts//options.entropy_interval <= options.window:
        raise FIGAROException("The number of entropy evaluations ({0}) must be greater than the window length ({1}).".format(n_pts//options.entropy_interval, options.window))
    
    # Reconstruction
    if not options.postprocess:
        if hier_flag:
            if options.run_events:
                mix = DPGMM(options.bounds)
                posteriors = []
                # Run each single-event analysis
                for i in tqdm(range(len(events)), desc = 'Events'):
                    ev     = events[i]
                    name_ev = names[i]
                    mix.initialise(prior_pars = get_priors(mix.bounds, samples = ev))
                    # Draw samples
                    draws = [mix.density_from_samples(ev) for _ in range(options.n_se_draws)]
                    posteriors.append(draws)
                    # Make plots
                    if options.save_single_event:
                        if dim == 1:
                            plot_median_cr(draws, samples = ev, out_folder = output_plots, name = name_ev, label = options.symbol, unit = options.unit, subfolder = True)
                        else:
                            plot_multidim(draws, samples = ev, out_folder = output_plots, name = name_ev, labels = symbols, units = units)
                    # Save single-event draws
                    save_density(draws, folder = output_draws, name = 'draws_'+name_ev, ext = options.ext)
                # Save all single-event draws together
                posteriors = np.array(posteriors)
                save_density(posteriors, folder = output_draws, name = 'posteriors_single_event', ext = options.ext)
            else:
                # Load pre-computed posteriors
                posteriors = load_density(Path(output_draws, 'posteriors_single_event.'+options.ext))
        # Actual analysis
        if not hier_flag:
            mix = DPGMM(options.bounds, prior_pars = get_priors(options.bounds, samples = samples, std = options.sigma_prior))
        else:
            mix = HDPGMM(options.bounds, prior_pars = get_priors(options.bounds, samples = all_samples, std = options.sigma_prior))
            samples = posteriors
        draws   = []
        entropy = []
        # This reproduces what it is done inside mix.density_from_samples while computing entropy for each new sample
        for j in tqdm(range(options.n_draws), desc = name, disable = (options.n_draws == 1)):
            S        = []
            n_eval_S = []
            mix.initialise()
            np.random.shuffle(samples)
            for i, s in tqdm(enumerate(samples), total = len(samples), disable = (j > 0)):
                mix.add_new_point(s)
                if i%options.entropy_interval == 0:
                    S.append(compute_entropy_single_draw(mix, n_draws = options.entropy_draws))
                    n_eval_S.append(i)
            draws.append(mix.build_mixture())
            entropy.append(S)
        draws     = np.array(draws)
        entropy   = np.concatenate(([n_eval_S], np.atleast_2d(entropy)))
        # Save reconstruction
        save_density(draws, options.output, name = 'draws_'+name, ext = options.ext)
        np.savetxt(Path(options.output, 'entropy_'+name+'.txt'), entropy)

    else:
        draws = load_density(Path(options.output, 'draws_'+name+'.'+options.ext))
        try:
            entropy = np.atleast_2d(np.loadtxt(Path(options.output, 'entropy_'+name+'.txt')))
        except FileNotFoundError:
            raise FileNotFoundError("No entropy_{0}.txt found. Please provide it or re-run the inference".format(name))

    if options.plot_dist:
        # Plot distribution
        if dim == 1:
            plot_median_cr(draws, injected = inj_density, samples = true_vals, out_folder = options.output, name = name, label = options.symbol, unit = options.unit, hierarchical = hier_flag)
        else:
            if options.symbol is not None:
                symbols = options.symbol.split(',')
            else:
                symbols = options.symbol
            if options.unit is not None:
                units = options.unit.split(',')
            else:
                units = options.unit
            plot_multidim(draws, samples = true_vals, out_folder = options.output, name = name, labels = symbols, units = units, hierarchical = hier_flag)

    n_samps_S = entropy[0]
    entropy   = entropy[1:]
    entropy_interval = int(n_samps_S[1]-n_samps_S[0])

    # Angular coefficients
    ang_coeff = np.atleast_2d([compute_angular_coefficients(S, options.window) for S in entropy])
    # Zero-crossings
    zero_crossings = [(options.window + np.where(np.diff(np.sign(ac)))[0])*entropy_interval for ac in ang_coeff]
    endpoints = []
    conv_not_reached_flag = False
    for zc in zero_crossings:
        try:
            endpoints.append(zc[options.zero_crossings])
        except IndexError:
            conv_not_reached_flag = True
    if not len(endpoints) == 0:
        EP = int(np.mean(endpoints))
        EP_label = '{0}'.format(EP) + '\ \mathrm{samples}'
        np.savetxt(Path(options.output, 'endpoint_'+name+'.txt'), np.atleast_1d(int(EP)))
        if not conv_not_reached_flag:
            print('Average number of samples required for convergence: {0}'.format(EP))
        else:
            print('Average number of samples required for convergence: {0}\nWARNING: at least one draw did not converge.'.format(EP))
    else:
        EP = None
        EP_label = None
        print('Convergence not reached yet')
    
    # Entropy & entropy derivative plot
    plot_1d_dist(n_samps_S, entropy, out_folder = options.output, name = 'entropy_'+name, label = 'N_{s}', median_label = '\mathrm{Entropy}')
    plot_1d_dist(np.linspace(options.window*entropy_interval, len(samples), len(ang_coeff[0]), dtype = int), ang_coeff, out_folder = options.output, name = 'ang_coeff_'+name, label = 'N_{s}', injected = np.zeros(len(ang_coeff[0])), true_value = EP, true_value_label = EP_label, median_label = '\mathrm{Entropy\ derivative}', injected_label = None)

if __name__ == '__main__':
    main()
