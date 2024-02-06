import numpy as np

import optparse as op
import dill
import importlib

from pathlib import Path
from tqdm import tqdm

from figaro.mixture import DPGMM, HDPGMM
from figaro.transform import transform_to_probit
from figaro.utils import save_options, load_options, get_priors
from figaro.plot import plot_median_cr, plot_multidim
from figaro.load import load_data, load_single_event, save_density, load_density

import ray
from ray.util import ActorPool

@ray.remote
class worker:
    def __init__(self, bounds,
                       out_folder_plots,
                       out_folder_draws,
                       ext         = 'pkl',
                       se_sigma    = None,
                       hier_sigma  = None,
                       scale       = None,
                       events      = None,
                       label       = None,
                       unit        = None,
                       save_se     = True,
                       MC_draws    = None,
                       probit      = True,
                       ):
        self.dim                  = bounds.shape[0]
        self.bounds               = bounds
        self.mixture              = DPGMM(self.bounds, probit = probit)
        self.hierarchical_mixture = HDPGMM(self.bounds,
                                           MC_draws   = MC_draws,
                                           probit     = probit,
                                           prior_pars = get_priors(self.bounds,
                                                                   samples      = events,
                                                                   std          = hier_sigma,
                                                                   scale        = scale,
                                                                   probit       = probit,
                                                                   hierarchical = True,
                                                                   )
                                            )
        self.out_folder_plots = out_folder_plots
        self.out_folder_draws = out_folder_draws
        self.se_sigma         = se_sigma
        self.scale            = scale
        self.save_se          = save_se
        self.label            = label
        self.unit             = unit
        self.posteriors       = None
        self.probit           = probit
        self.ext              = ext

    def run_event(self, pars):
        # Unpack data
        samples, name, n_draws = pars
        # Copying (issues with shuffling)
        ev = np.copy(samples)
        ev.setflags(write = True)
        # Actual inference
        prior_pars = get_priors(self.bounds, samples = ev, probit = self.probit, std = self.se_sigma, scale = self.scale, hierarchical = False)
        self.mixture.initialise(prior_pars = prior_pars)
        draws      = [self.mixture.density_from_samples(ev) for _ in range(n_draws)]
        # Plots
        plt_bounds = np.atleast_2d([ev.min(axis = 0), ev.max(axis = 0)]).T
        if self.save_se:
            if self.dim == 1:
                plot_median_cr(draws,
                               samples    = ev,
                               bounds     = plt_bounds[0],
                               out_folder = self.out_folder_plots,
                               name       = name,
                               label      = self.label,
                               unit       = self.unit,
                               subfolder  = True,
                               )
            else:
                plot_multidim(draws,
                              samples    = ev,
                              bounds     = plt_bounds,
                              out_folder = self.out_folder_plots,
                              name       = name,
                              labels     = self.label,
                              units      = self.unit,
                              subfolder  = True,
                              )
        # Saving
        save_density(draws, folder = self.out_folder_draws, name = 'draws_'+name, ext = self.ext)
        return draws

    def draw_hierarchical(self):
        return self.hierarchical_mixture.density_from_samples(self.posteriors)
    
    def load_posteriors(self, posteriors):
        self.posteriors = np.copy(posteriors)
        self.posteriors.setflags(write = True)
        for i in range(len(self.posteriors)):
            self.posteriors[i].setflags(write = True)
            
def main():

    parser = op.OptionParser()
    # Input/output
    parser.add_option("-i", "--input", type = "string", dest = "samples_folder", help = "Folder with single-event samples files", default = None)
    parser.add_option("-b", "--bounds", type = "string", dest = "bounds", help = "Density bounds. Must be a string formatted as '[[xmin, xmax], [ymin, ymax],...]'. For 1D distributions use '[xmin, xmax]'. Quotation marks are required and scientific notation is accepted", default = None)
    parser.add_option("-o", "--output", type = "string", dest = "output", help = "Output folder. Default: same directory as samples folder", default = None)
    parser.add_option("-j", dest = "json", action = 'store_true', help = "Save mixtures in json file", default = False)
    parser.add_option("--inj_density", type = "string", dest = "inj_density_file", help = "Python module with injected density - please name the method 'density'", default = None)
    parser.add_option("--selfunc", type = "string", dest = "selfunc_file", help = "Python module with selection function - please name the method 'selection_function'", default = None)
    parser.add_option("--parameter", type = "string", dest = "par", help = "GW parameter(s) to be read from files", default = None)
    parser.add_option("--waveform", type = "string", dest = "wf", help = "Waveform to load from samples file. To be used in combination with --parameter. Accepted values: 'combined', 'imr', 'seob'", default = 'combined')
    # Plot
    parser.add_option("--name", type = "string", dest = "h_name", help = "Name to be given to hierarchical inference files. Default: same name as samples folder parent directory", default = None)
    parser.add_option("-p", "--postprocess", dest = "postprocess", action = 'store_true', help = "Postprocessing", default = False)
    parser.add_option("-s", "--save_se", dest = "save_single_event", action = 'store_true', help = "Save single event plots", default = False)
    parser.add_option("--symbol", type = "string", dest = "symbol", help = "LaTeX-style quantity symbol, for plotting purposes", default = None)
    parser.add_option("--unit", type = "string", dest = "unit", help = "LaTeX-style quantity unit, for plotting purposes", default = None)
    parser.add_option("--hier_samples", type = "string", dest = "true_vals", help = "Samples from hierarchical distribution (true single-event values, for simulations only)", default = None)
    # Settings
    parser.add_option("--draws", type = "int", dest = "n_draws", help = "Number of draws for hierarchical distribution", default = 100)
    parser.add_option("--se_draws", type = "int", dest = "n_se_draws", help = "Number of draws for single-event distribution. Default: same as hierarchical distribution", default = None)
    parser.add_option("--n_samples_dsp", type = "int", dest = "n_samples_dsp", help = "Number of samples to analyse (downsampling). Default: all", default = -1)
    parser.add_option("--exclude_points", dest = "exclude_points", action = 'store_true', help = "Exclude points outside bounds from analysis", default = False)
    parser.add_option("--cosmology", type = "string", dest = "cosmology", help = "Cosmological parameters (h, om, ol). Default values from Planck (2021)", default = '0.674,0.315,0.685')
    parser.add_option("-e", "--events", dest = "run_events", action = 'store_false', help = "Skip single-event analysis", default = True)
    parser.add_option("--se_sigma_prior", dest = "se_sigma_prior", type = "string", help = "Expected standard deviation (prior) for single-event inference - single value or n-dim values. If None, it is estimated from samples", default = None)
    parser.add_option("--sigma_prior", dest = "sigma_prior", type = "string", help = "Expected standard deviation (prior) for hierarchical inference - single value or n-dim values. If None, it is estimated from samples", default = None)
    parser.add_option("--fraction", dest = "scale", type = "float", help = "Fraction of samples standard deviation for sigma prior. Overrided by sigma_prior.", default = None)
    parser.add_option("--n_parallel", dest = "n_parallel", type = "int", help = "Number of parallel threads", default = 4)
    parser.add_option("--mc_draws", dest = "mc_draws", type = "int", help = "Number of draws for assignment MC integral", default = None)
    parser.add_option("--snr_threshold", dest = "snr_threshold", type = "float", help = "SNR threshold for simulated GW datasets", default = None)
    parser.add_option("--far_threshold", dest = "far_threshold", type = "float", help = "FAR threshold for simulated GW datasets", default = None)
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
    # Read hierarchical name
    if options.h_name is None:
        options.h_name = options.output.parts[-1]
    # File extension
    if options.json:
        options.ext = 'json'
    else:
        options.ext = 'pkl'

    if options.config is not None:
        load_options(options, options.config)
    save_options(options, options.output, name = options.h_name)

    # Read bounds
    if options.bounds is not None:
        options.bounds = np.array(np.atleast_2d(eval(options.bounds)), dtype = np.float64)
    elif options.bounds is None and not options.postprocess:
        raise Exception("Please provide bounds for the inference (use -b '[[xmin,xmax],[ymin,ymax],...]')")

    # Read cosmology
    options.h, options.om, options.ol = (float(x) for x in options.cosmology.split(','))
    # Read parameter(s)
    if options.par is not None:
        options.par = options.par.split(',')
    # Read number of single-event draws
    if options.n_se_draws is None:
        options.n_se_draws = options.n_draws
    if options.se_sigma_prior is not None:
        options.se_sigma_prior = np.array([float(s) for s in options.se_sigma_prior.split(',')])
    if options.sigma_prior is not None:
        options.sigma_prior = np.array([float(s) for s in options.sigma_prior.split(',')])

    # If provided, load injected density
    inj_density = None
    if options.inj_density_file is not None:
        inj_file_name = Path(options.inj_density_file).parts[-1].split('.')[0]
        spec = importlib.util.spec_from_file_location(inj_file_name, options.inj_density_file)
        inj_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inj_module)
        inj_density = inj_module.density
    #If provided, load selecton function
    selfunc = None
    if options.selfunc_file is not None:
        selfunc_file_name = Path(options.selfunc_file).parts[-1].split('.')[0]
        spec = importlib.util.spec_from_file_location(selfunc_file_name, options.selfunc_file)
        selfunc_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(selfunc_module)
        selfunc = selfunc_module.selection_function
    # If provided, load true values
    true_vals = None
    if options.true_vals is not None:
        options.true_vals = Path(options.true_vals).resolve()
        true_vals, true_name = load_single_event(options.true_vals, par = options.par, h = options.h, om = options.om, ol = options.ol, waveform = options.wf)
        if np.shape(true_vals)[-1] == 1:
            true_vals = true_vals.flatten()
    # Load samples
    events, names = load_data(options.samples_folder, par = options.par, n_samples = options.n_samples_dsp, h = options.h, om = options.om, ol = options.ol, waveform = options.wf, snr_threshold = options.snr_threshold, far_threshold = options.far_threshold)
    try:
        dim = np.shape(events[0][0])[-1]
    except IndexError:
        dim = 1
    if options.exclude_points:
        print("Ignoring points outside bounds.")
        for i, ev in enumerate(events):
            events[i] = ev[np.where((np.prod(options.bounds[:,0] < ev, axis = 1) & np.prod(ev < options.bounds[:,1], axis = 1)))]
    else:
        # Check if all samples are within bounds
        all_samples = np.atleast_2d(np.concatenate(events))
        if options.probit:
            if not np.all([(all_samples[:,i] > options.bounds[i,0]).all() and (all_samples[:,i] < options.bounds[i,1]).all() for i in range(dim)]):
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
    else:
        symbols = options.symbol
        units   = options.unit
    
    # Reconstruction
    if not options.postprocess:
        ray.init(num_cpus = options.n_parallel)
        pool = ActorPool([worker.remote(bounds           = options.bounds,
                                        out_folder_plots = output_plots,
                                        out_folder_draws = output_draws,
                                        ext              = options.ext,
                                        se_sigma         = options.se_sigma_prior,
                                        hier_sigma       = options.sigma_prior,
                                        scale            = options.scale,
                                        events           = events,
                                        label            = symbols,
                                        unit             = units,
                                        save_se          = options.save_single_event,
                                        MC_draws         = options.mc_draws,
                                        probit           = options.probit,
                                        )
                          for _ in range(options.n_parallel)])
        
        if options.run_events:
            # Run each single-event analysis
            posteriors = []
            for s in tqdm(pool.map_unordered(lambda a, v: a.run_event.remote(v), [[ev, name, options.n_se_draws] for ev, name in zip(events, names)]), total = len(events), desc = 'Events'):
                posteriors.append(s)
            # Save all single-event draws together
            posteriors = np.array(posteriors)
            save_density(posteriors, folder = output_draws, name = 'posteriors_single_event', ext = options.ext)
        else:
            posteriors = load_density(Path(output_draws, 'posteriors_single_event.'+options.ext))
        # Load posteriors
        for s in pool.map(lambda a, v: a.load_posteriors.remote(v), [posteriors for _ in range(options.n_parallel)]):
            pass
        # Run hierarchical analysis
        draws = []
        for s in tqdm(pool.map_unordered(lambda a, v: a.draw_hierarchical.remote(), [_ for _ in range(options.n_draws)]), total = options.n_draws, desc = 'Sampling'):
            draws.append(s)
        draws = np.array(draws)
        # Save draws
        save_density(draws, folder = output_draws, name = 'draws_'+options.h_name, ext = options.ext)
    else:
        draws = load_density(Path(output_draws, 'draws_'+options.h_name+'.'+options.ext))
    # Plot
    if dim == 1:
        plot_median_cr(draws,
                       injected     = inj_density,
                       selfunc      = selfunc,
                       samples      = true_vals,
                       out_folder   = output_plots,
                       name         = options.h_name,
                       label        = options.symbol,
                       unit         = options.unit,
                       hierarchical = True,
                       )
    else:
        plot_multidim(draws,
                      samples      = true_vals,
                      out_folder   = output_plots,
                      name         = options.h_name,
                      labels       = symbols,
                      units        = units,
                      hierarchical = True,
                      )

if __name__ == '__main__':
    main()
