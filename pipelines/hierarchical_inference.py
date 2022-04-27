import numpy as np

import optparse as op
import configparser
import json
import dill
import importlib

from pathlib import Path
from tqdm import tqdm

from figaro.mixture import DPGMM, HDPGMM
from figaro.transform import transform_to_probit
from figaro.utils import save_options, plot_median_cr, plot_multidim
from figaro.load import load_data

def main():

    parser = op.OptionParser()
    # Input/output
    parser.add_option("-i", "--input", type = "string", dest = "samples_folder", help = "Folder with single-event samples files")
    parser.add_option("-b", "--bounds", type = "string", dest = "bounds", help = "Density bounds. Must be a string formatted as '[[xmin, xmax], [ymin, ymax],...]'. For 1D distributions use '[[xmin, xmax]]'. Quotation marks are required and scientific notation is accepted", default = None)
    parser.add_option("-o", "--output", type = "string", dest = "output", help = "Output folder. Default: same directory as samples folder", default = None)
    parser.add_option("--inj_density", type = "string", dest = "inj_density_file", help = "Python module with injected density - please name the method 'density'", default = None)
    parser.add_option("--parameter", type = "string", dest = "par", help = "GW parameter(s) to be read from files", default = 'm1')
    # Plot
    parser.add_option("--name", type = "string", dest = "h_name", help = "Name to be given to hierarchical inference files. Default: same name as samples folder parent directory", default = None)
    parser.add_option("-p", "--postprocess", dest = "postprocess", action = 'store_true', help = "Postprocessing", default = False)
    parser.add_option("--symbol", type = "string", dest = "symbol", help = "LaTeX-style quantity symbol, for plotting purposes", default = None)
    parser.add_option("--unit", type = "string", dest = "unit", help = "LaTeX-style quantity unit, for plotting purposes", default = None)
    parser.add_option("--hier_samples", type = "string", dest = "true_vals", help = "Samples from hierarchical distribution (true single-event values, for simulations only)", default = None)
    # Settings
    parser.add_option("--draws", type = "int", dest = "n_draws", help = "Number of draws for hierarchical distribution", default = 100)
    parser.add_option("--se_draws", type = "int", dest = "n_se_draws", help = "Number of draws for single-event distribution. Default: same as hierarchical distribution", default = None)
    parser.add_option("--n_samples_dsp", type = "int", dest = "n_samples_dsp", help = "Number of samples to analyse (downsampling). Default: all", default = -1)
    parser.add_option("--cosmology", type = "string", dest = "cosmology", help = "Cosmological parameters (h, om, ol). Default values from Planck (2021)", default = '0.674,0.315,0.685')
    parser.add_option("-e", "--events", dest = "run_events", action = 'store_false', help = "Skip single-event analysis", default = True)

    (options, args) = parser.parse_args()

    # Paths
    options.samples_folder = Path(options.samples_folder).resolve()
    if options.output is not None:
        options.output = Path(options.output).resolve()
        if not options.output.exists():
            options.output.mkdir(parents=True)
    else:
        options.output = options.samples_folder.parent
    output_plots = Path(options.output, 'plots')
    if not output_plots.exists():
        output_plots.mkdir()
    output_pkl = Path(options.output, 'draws')
    if not output_pkl.exists():
        output_pkl.mkdir()
    # Read hierarchical name
    if options.h_name is None:
        options.h_name = options.samples_folder.parent.parts[-1]
    # Read bounds
    if options.bounds is not None:
        options.bounds = np.array(json.loads(options.bounds))
    elif options.bounds is None and not options.postprocess:
        print("Please provide bounds for the inference (use -b '[[xmin,xmax],[ymin,ymax],...]')")
        exit()
    # If provided, load injected density
    inj_density = None
    if options.inj_density_file is not None:
        inj_file_name = options.inj_density_file.split('/')[-1].split('.')[0]
        spec = importlib.util.spec_from_file_location(inj_file_name, options.inj_density_file)
        inj_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inj_module)
        inj_density = inj_module.density
    # If provided, load true values
    true_vals = None
    if options.true_vals is not None:
        options.true_vals = Path(options.true_vals).resolve()
        true_vals = np.loadtxt(options.true_vals)
    # Read cosmology
    options.h, options.om, options.ol = (float(x) for x in options.cosmology.split(','))
    # Read parameter(s)
    options.par = options.par.split(',')
    # Read number of single-event draws
    if options.n_se_draws is None:
        options.n_se_draws = options.n_draws
    
    save_options(options)
    
    # Load samples
    events, names = load_data(options.samples_folder, par = options.par, n_samples = options.n_samples_dsp, h = options.h, om = options.om, ol = options.ol)
    all_samples = np.atleast_2d(np.concatenate(events)).T
    try:
        dim = np.shape(events[0][0])[-1]
    except IndexError:
        dim = 1
    # Plot labels
    if dim > 1:
        if options.symbol is not None:
            symbols = options.symbol.split(',')
        else:
            symbols = options.symbol
        if options.unit is not None:
            units = options.unit.split(',')
        else:
            units = options.unit
    
    # Reconstruction
    if not options.postprocess:
        if options.run_events:
            mix = DPGMM(options.bounds)
            posteriors = []
            # Run each single-event analysis
            for i in tqdm(range(len(events)), desc = 'Events'):
                ev   = events[i]
                name = names[i]
                # Variance prior from samples
                probit_samples = transform_to_probit(ev, options.bounds)
                sigma = np.atleast_2d(np.var(probit_samples, axis = 0)/25)
                mix.initialise(prior_pars = (1e-1, sigma/25, dim, np.zeros(dim)))
                # Draw samples
                draws = []
                for _ in range(options.n_se_draws):
                    np.random.shuffle(ev)
                    mix.density_from_samples(ev)
                    draws.append(mix.build_mixture())
                    n = np.random.uniform(1.5, 8)
                    mix.initialise(prior_pars = (1e-1, sigma/n**2, dim, np.zeros(dim)))
                posteriors.append(draws)
                # Make plots
                if dim == 1:
                    plot_median_cr(draws, injected = inj_density, samples = ev, out_folder = output_plots, name = name, label = options.symbol, unit = options.unit)
                else:
                    plot_multidim(draws, dim, samples = ev, out_folder = output_plots, name = name, labels = symbols, units = units)
                # Save single-event draws
                with open(Path(output_pkl, 'draws_'+name+'.pkl'), 'wb') as f:
                    dill.dump(np.array(draws), f)
            # Save all single-event draws together
            posteriors = np.array(posteriors)
            with open(Path(output_pkl, 'posteriors_single_event.pkl'), 'wb') as f:
                dill.dump(posteriors, f)
        else:
            # Load pre-computed posteriors
            try:
                with open(Path(output_pkl, 'posteriors_single_event.pkl'), 'rb') as f:
                    posteriors = dill.load(f)
            except FileNotFoundError:
                print("No posteriors_single_event.pkl file found. Please provide it or re-run the single-event inference")
                exit()
        probit_samples = transform_to_probit(all_samples, options.bounds)
        sigma = np.atleast_2d(np.var(probit_samples, axis = 0))
        mix = HDPGMM(options.bounds, prior_pars = (1e-1, sigma/25, dim, np.zeros(dim)))
        draws = []
        # Run hierarchical analysis
        for _ in tqdm(range(options.n_draws), desc = 'Hierarchical'):
            np.random.shuffle(posteriors)
            mix.density_from_samples(posteriors)
            draws.append(mix.build_mixture())
            n = np.random.uniform(1.5, 8)
            mix.initialise(prior_pars = (1e-1, sigma/n**2, dim, np.zeros(dim)))
        draws = np.array(draws)
        with open(Path(output_pkl, 'draws_'+options.h_name+'.pkl'), 'wb') as f:
            dill.dump(draws, f)
    else:
        try:
            with open(Path(output_pkl, 'draws_'+options.h_name+'.pkl'), 'rb') as f:
                draws = dill.load(f)
        except FileNotFoundError:
            print("No draws_{0}.pkl file found. Please provide it or re-run the inference".format(options.h_name))
            exit()
    # Plot
    if dim == 1:
        plot_median_cr(draws, injected = inj_density, samples = true_vals, out_folder = output_plots, name = options.h_name, label = options.symbol, unit = options.unit)
    else:
        plot_multidim(draws, dim, samples = true_vals, out_folder = output_plots, name = options.h_name, labels = symbols, units = units)

if __name__ == '__main__':
    main()
