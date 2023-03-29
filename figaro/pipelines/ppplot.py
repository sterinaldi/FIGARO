import numpy as np

import optparse as op
import json
import dill
import importlib

from pathlib import Path
from tqdm import tqdm

from figaro.mixture import DPGMM
from figaro.transform import transform_to_probit
from figaro.utils import save_options, load_options, recursive_grid, get_priors
from figaro.plot import plot_median_cr, plot_multidim, pp_plot_levels
from figaro.credible_regions import FindLevelForHeight, FindNearest_Grid
from figaro.load import load_data, save_density, load_density

def main():

    parser = op.OptionParser()
    # Input/output
    parser.add_option("-i", "--input", type = "string", dest = "samples_folder", help = "Folder with single-event samples files. .txt files only", default = None)
    parser.add_option("-b", "--bounds", type = "string", dest = "bounds", help = "Density bounds. Must be a string formatted as '[[xmin, xmax], [ymin, ymax],...]'. For 1D distributions use '[xmin, xmax]'. Quotation marks are required and scientific notation is accepted", default = None)
    parser.add_option("--true_vals", type = "string", dest = "true_vals", help = "JSON file storing a dictionary with the injected values. Dictionary keys must match single-event samples files names", default = None)
    parser.add_option("-o", "--output", type = "string", dest = "output", help = "Output folder. Default: same directory as samples folder", default = None)
    parser.add_option("-j", dest = "json", action = 'store_true', help = "Save mixtures in json file", default = False)
    # Plot
    parser.add_option("--name", type = "string", dest = "name", help = "Name to be given to pp-plot files. Default: MDC", default = 'MDC')
    parser.add_option("-p", "--postprocess", dest = "postprocess", action = 'store_true', help = "Postprocessing", default = False)
    parser.add_option("-s", "--save_plot", dest = "save_plots", action = 'store_true', help = "Save single event plots", default = False)
    parser.add_option("--symbol", type = "string", dest = "symbol", help = "LaTeX-style quantity symbol, for plotting purposes", default = None)
    parser.add_option("--unit", type = "string", dest = "unit", help = "LaTeX-style quantity unit, for plotting purposes", default = None)
    # Settings
    parser.add_option("--draws", type = "int", dest = "n_draws", help = "Number of draws for each single-event distribution", default = 100)
    parser.add_option("--n_samples_dsp", type = "int", dest = "n_samples_dsp", help = "Number of samples to analyse (downsampling). Default: all", default = -1)
    parser.add_option("--exclude_points", dest = "exclude_points", action = 'store_true', help = "Exclude points outside bounds from analysis", default = False)
    parser.add_option("--grid_points", dest = "grid_points", type = "string", help = "Number of grid points for each dimension. Single integer or one int per dimension", default = None)
    parser.add_option("-e", "--events", dest = "run_events", action = 'store_false', help = "Skip single-event analysis", default = True)
    parser.add_option("--no_probit", dest = "probit", action = 'store_false', help = "Disable probit transformation", default = True)
    parser.add_option("--config", dest = "config", type = "string", help = "Config file. Warning: command line options are ignored if provided", default = None)

    (options, args) = parser.parse_args()

    # Paths
    if options.samples_folder is not None:
        options.samples_folder = Path(options.samples_folder).resolve()
    elif options.config is not None:
        options.samples_folder = Path('.').resolve()
    else:
        raise Exception("Please provide path to samples.")
    if options.output is not None:
        options.output = Path(options.output).resolve()
        if not options.output.exists():
            options.output.mkdir(parents=True)
    else:
        options.output = options.samples_folder.parent
    if options.config is not None:
        options.config = Path(options.config).resolve()
    output_plots = Path(options.output, 'plots')
    if not output_plots.exists():
        output_plots.mkdir()
    output_draws = Path(options.output, 'draws')
    if not output_draws.exists():
        output_draws.mkdir()

    # File extension
    if options.json:
        options.ext = 'json'
    else:
        options.ext = 'pkl'
    # Check that all files are .txt
    if not (np.array([f.suffix for f in options.samples_folder.glob('*')]) == '.txt').all():
        raise Exception("Only .txt files are currently supported for PP-plot analysis")

    if options.config is not None:
        load_options(options, options.config)
    save_options(options, options.output)

    # Read bounds
    if options.bounds is not None:
        options.bounds = np.array(np.atleast_2d(eval(options.bounds)), dtype = np.float64)
    elif options.bounds is None and not options.postprocess:
        raise Exception("Please provide bounds for the inference (use -b '[[xmin,xmax],[ymin,ymax],...]')")
    else:
        raise Exception("Please provide JSON file with true values")

    # Load samples
    events, names = load_data(options.samples_folder, n_samples = options.n_samples_dsp)
    try:
        dim = np.shape(events[0][0])[-1]
    except IndexError:
        dim = 1
    if dim > 3:
        raise Exception("PP-plots can be computed up to 3 dimensions")
    # Load true values
    if options.true_vals is not None:
        options.true_vals = Path(options.true_vals).resolve()
        with open(options.true_vals, 'r') as f:
            true_vals = json.load(f)
    # Check all events have an entry in true_vals dict
    if not np.array([name in true_vals.keys() for name in names]).all():
        raise Exception("Please provide a dictionary storing all the true values. Dict keys must match event names. The following events appear not to have a true value:\n{0}".format(np.array(names)[np.where([name not in true_vals.keys() for name in names])]))

    if options.exclude_points:
        print("Ignoring points outside bounds.")
        for i, ev in enumerate(events):
            events[i] = ev[np.where((np.prod(options.bounds[:,0] < ev, axis = 1) & np.prod(ev < options.bounds[:,1], axis = 1)))]
        all_samples = np.atleast_2d(np.concatenate(events))
    else:
        # Check if all samples are within bounds
        all_samples = np.atleast_2d(np.concatenate(events))
        if options.probit:
            if not np.alltrue([(all_samples[:,i] > options.bounds[i,0]).all() and (all_samples[:,i] < options.bounds[i,1]).all() for i in range(dim)]):
                raise ValueError("One or more samples are outside the given bounds.")
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
    # Read grid points
    if options.grid_points is not None:
        pts = options.grid_points.split(',')
        if len(pts) == dim:
            options.grid_points = np.array([int(d) for d in pts])
        elif len(pts) == 1:
            options.grid_points = np.ones(dim, dtype = int)*int(pts[0])
        else:
            print("Wrong number of grid point provided. Falling back to default number")
            options.grid_points = np.ones(dim, dtype = int)*200
    else:
        options.grid_points = np.ones(dim, dtype = int)*200

    # Reconstruction
    if not options.postprocess:
        mix        = DPGMM(options.bounds, probit = options.probit)
        grid, diff = recursive_grid(options.bounds, options.grid_points)
        logdiff    = np.sum(np.log(diff))
        posteriors = []
        CR_levels  = []
        CR_medians = []
        # Run single-event analysis
        for i in tqdm(range(len(events)), desc = 'Events'):
            ev   = events[i]
            name = names[i]
            # Load true value
            t = true_vals[name]
            if options.run_events:
                # Estimate prior pars from samples
                mix.initialise(prior_pars = get_priors(mix.bounds, samples = ev, probit = options.probit, hierarchical = False))
                # Draw samples
                draws = [mix.density_from_samples(ev) for _ in range(options.n_draws)]
                posteriors.append(draws)
                if options.save_plots:
                    if dim == 1:
                        plot_median_cr(draws,
                                       samples    = ev,
                                       out_folder = output_plots,
                                       name       = name,
                                       label      = options.symbol,
                                       unit       = options.unit,
                                       subfolder  = True,
                                       true_value = t,
                                       )
                    else:
                        plot_multidim(draws,
                                      samples    = ev,
                                      out_folder = output_plots,
                                      name       = name,
                                      labels     = symbols,
                                      units      = units,
                                      true_value = t,
                                      )
                # Save single-event draws
                save_density(draws, folder = output_draws, name = 'draws_'+name, ext = options.ext)
            else:
                draws = load_density(Path(output_draws, 'draws_'+name+'.'+options.ext))
            # Evaluate mixtures
            logP      = np.array([d.logpdf(grid) for d in draws])
            # Find true_value
            true_idx  = FindNearest_Grid(grid, t)
            logP_true = np.array([logP_i[true_idx] for logP_i in logP])
            # Compute credible levels
            CR = np.array([FindLevelForHeight(logP_i, logP_i_true, logdiff) for logP_i, logP_i_true in zip(logP, logP_true)])
            CR_medians.append(np.median(CR))
            CR_levels.append(CR)
            
        # Save all single-event draws together (might be useful for future hierarchical analysis)
        posteriors = np.array(posteriors)
        with open(Path(output_draws, 'posteriors_single_event.pkl'), 'wb') as f:
            dill.dump(posteriors, f)
        # Save credible levels
        CR_levels  = np.array(CR_levels)
        CR_medians = np.array(CR_medians)
        np.savetxt(Path(options.output, 'CR_levels.txt'), CR_levels, header = 'CR levels for events\nEach row is a different event, columns represent different draws')
        np.savetxt(Path(options.output, 'CR_medians.txt'), CR_medians)
    else:
        CR_levels = np.genfromtxt(Path(options.output, 'CR_levels.txt'))
        if len(CR_levels.shape) == 1:
            CR_levels = np.atleast_2d(CR_levels).T
        CR_medians = np.genfromtxt(Path(options.output, 'CR_medians.txt'))

    pp_plot_levels(CR_levels, median_CR = CR_medians, out_folder = options.output, name = options.name)

if __name__ == '__main__':
    main()
