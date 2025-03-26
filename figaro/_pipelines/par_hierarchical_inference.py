#!/usr/bin/env python3
import numpy as np

import optparse
import importlib
import warnings

from pathlib import Path
from tqdm import tqdm

from figaro.mixture import DPGMM, HDPGMM
from figaro.utils import save_options, load_options, get_priors
from figaro.plot import plot_median_cr, plot_multidim
from figaro.load import load_data, load_single_event, load_selection_function, save_density, load_density, supported_pars
from figaro.rate import sample_rate, normalise_alpha_factor, plot_integrated_rate, plot_differential_rate
from figaro.cosmology import _decorator_dVdz, dVdz_approx_planck18, dVdz_approx_planck15
from figaro.marginal import marginalise

import ray
from ray.util import ActorPool

@ray.remote
class worker:
    def __init__(self, bounds,
                       out_folder_plots,
                       out_folder_draws,
                       ext         = 'json',
                       se_sigma    = None,
                       hier_sigma  = None,
                       scale       = None,
                       events      = None,
                       label       = None,
                       unit        = None,
                       save_se     = True,
                       MC_draws    = None,
                       probit      = True,
                       selfunc     = None,
                       inj_pdf     = None,
                       n_total_inj = None,
                       ):
        self.dim                  = bounds.shape[0]
        self.bounds               = bounds
        self.mixture              = DPGMM(self.bounds, probit = probit)
        self.hierarchical_mixture = HDPGMM(self.bounds,
                                           MC_draws           = MC_draws,
                                           probit             = probit,
                                           selection_function = selfunc,
                                           injection_pdf      = inj_pdf,
                                           total_injections   = n_total_inj,
                                           prior_pars         = get_priors(self.bounds,
                                                                           samples      = events,
                                                                           std          = hier_sigma,
                                                                           scale        = scale,
                                                                           probit       = probit,
                                                                           hierarchical = True,
                                                                           ),
                                           )
        self.out_folder_plots = out_folder_plots
        self.out_folder_draws = out_folder_draws
        self.se_sigma         = se_sigma
        self.scale            = scale
        self.save_se          = save_se
        self.label            = label
        self.unit             = unit
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
        draws      = [self.mixture.density_from_samples(ev, make_comp = False) for _ in range(n_draws)]
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
        return self.hierarchical_mixture.density_from_samples(self.posteriors, make_comp = False)
    
    def load_posteriors(self, posteriors):
        self.posteriors = np.copy(posteriors)
        self.posteriors.setflags(write = True)
        for i in range(len(self.posteriors)):
            self.posteriors[i].setflags(write = True)
            
def main():

    parser = optparse.OptionParser(prog = 'figaro-par-hierarchical', description = 'Parallelised hierarchical probability density reconstruction')
    # Input/output
    parser.add_option("-i", "--input", type = "string", dest = "input", help = "Folder with single-event samples files", default = None)
    parser.add_option("-b", "--bounds", type = "string", dest = "bounds", help = "Density bounds. Must be a string formatted as '[[xmin, xmax], [ymin, ymax],...]'. For 1D distributions use '[xmin, xmax]'. Quotation marks are required and scientific notation is accepted", default = None)
    parser.add_option("-o", "--output", type = "string", dest = "output", help = "Output folder. Default: same directory as samples folder", default = None)
    parser.add_option("--ext", dest = "ext", type = "choice", choices = ['pkl', 'json'], help = "Format of mixture output file", default = 'json')
    parser.add_option("--inj_density", type = "string", dest = "inj_density_file", help = "Python module with injected density - please name the method 'density'", default = None)
    parser.add_option("--selfunc", type = "string", dest = "selfunc_file", help = "Python module with selection function - please name the method 'selection_function'", default = None)
    parser.add_option("--parameter", type = "string", dest = "par", help = "GW parameter(s) to be read from file", default = None)
    parser.add_option("--waveform", type = "choice", dest = "wf", help = "Waveform to load from samples file. To be used in combination with --parameter.", choices = ['combined', 'seob', 'imr'], default = 'combined')
    # Plot
    parser.add_option("--name", type = "string", dest = "hier_name", help = "Name to be given to hierarchical inference files. Default: same name as samples folder parent directory", default = None)
    parser.add_option("-p", "--postprocess", dest = "postprocess", action = 'store_true', help = "Postprocessing", default = False)
    parser.add_option("-s", "--save_se", dest = "save_single_event", action = 'store_true', help = "Save single event plots", default = False)
    parser.add_option("--symbol", type = "string", dest = "symbol", help = "LaTeX-style quantity symbol, for plotting purposes", default = None)
    parser.add_option("--unit", type = "string", dest = "unit", help = "LaTeX-style quantity unit, for plotting purposes", default = None)
    parser.add_option("--hier_samples", type = "string", dest = "hier_samples", help = "Samples from hierarchical distribution (true single-event values, for simulations only)", default = None)
    # Settings
    parser.add_option("--draws", type = "int", dest = "draws", help = "Number of draws for hierarchical distribution", default = 100)
    parser.add_option("--se_draws", type = "int", dest = "se_draws", help = "Number of draws for single-event distribution. Default: same as hierarchical distribution", default = None)
    parser.add_option("--n_samples_dsp", type = "int", dest = "n_samples_dsp", help = "Number of samples to analyse (downsampling). Default: all", default = -1)
    parser.add_option("--exclude_points", dest = "exclude_points", action = 'store_true', help = "Exclude points outside bounds from analysis", default = False)
    parser.add_option("--cosmology", type = "choice", dest = "cosmology", help = "Set of cosmological parameters. Default values from Planck (2021)", choices = ['Planck18', 'Planck15'], default = 'Planck18')
    parser.add_option("-e", "--events", dest = "run_events", action = 'store_false', help = "Skip single-event analysis", default = True)
    parser.add_option("--se_sigma_prior", dest = "se_sigma_prior", type = "string", help = "Expected standard deviation (prior) for single-event inference - single value or n-dim values. If None, it is estimated from samples", default = None)
    parser.add_option("--sigma_prior", dest = "sigma_prior", type = "string", help = "Expected standard deviation (prior) for hierarchical inference - single value or n-dim values. If None, it is estimated from samples", default = None)
    parser.add_option("--fraction", dest = "fraction", type = "float", help = "Fraction of samples standard deviation for sigma prior. Overrided by sigma_prior.", default = None)
    parser.add_option("--n_parallel", dest = "n_parallel", type = "int", help = "Number of parallel threads", default = 2)
    parser.add_option("--mc_draws", dest = "mc_draws", type = "int", help = "Number of draws for assignment MC integral", default = None)
    parser.add_option("--far_threshold", dest = "far_threshold", type = "float", help = "FAR threshold for LVK sensitivity estimate injections", default = 1.)
    parser.add_option("--snr_threshold", dest = "snr_threshold", type = "float", help = "SNR threshold for LVK sensitivity estimate injections", default = 10.)
    parser.add_option("--no_probit", dest = "probit", action = 'store_false', help = "Disable probit transformation", default = True)
    parser.add_option("--config", dest = "config", type = "string", help = "Config file. Warning: command line options override config options", default = None)
    parser.add_option("--rate", dest = "rate", action = 'store_true', help = "Compute rate", default = False)
    parser.add_option("--include_dvdz", dest = "include_dvdz", action = 'store_true', help = "Include dV/dz*(1+z)^{-1} term in selection effects.", default = False)
    parser.add_option("-l", "--likelihood", dest = "likelihood", action = 'store_true', help = "Resample posteriors to get likelihood samples (only for GW data)", default = False)

    (options, args) = parser.parse_args()

    if options.config is not None:
        options = load_options(options, parser)
    # Paths
    if options.input is not None:
        options.input = Path(options.input).resolve()
    elif options.config is not None:
        options.input = Path('.').resolve()
    else:
        raise Exception("Please provide path to samples.")
    if options.output is not None:
        options.output = Path(options.output).resolve()
        if not options.output.exists():
            options.output.mkdir(parents=True)
    else:
        options.output = options.input.parent
    if options.config is not None:
        options.config = Path(options.config).resolve()
    output_plots = Path(options.output, 'plots')
    if not output_plots.exists():
        output_plots.mkdir()
    output_draws = Path(options.output, 'draws')
    if not output_draws.exists():
        output_draws.mkdir()
    if options.rate:
        output_rate = Path(options.output, 'rate')
        if not output_rate.exists():
            output_rate.mkdir()
    # Read hierarchical name
    if options.hier_name is None:
        options.hier_name = options.output.parts[-1]
    if options.selfunc_file is None:
        hier_name = 'observed_'+options.hier_name
    else:
        hier_name = 'intrinsic_'+options.hier_name

    if options.config is None:
        save_options(options, options.output, name = options.hier_name)

    # Read bounds
    if options.bounds is not None:
        options.bounds = np.array(np.atleast_2d(eval(options.bounds)), dtype = np.float64)
    elif options.bounds is None and not options.postprocess:
        raise Exception("Please provide bounds for the inference (use -b '[[xmin,xmax],[ymin,ymax],...]')")
    # Read parameter(s)
    if options.par is not None:
        options.par = options.par.split(',')
        if not np.all([par in supported_pars for par in options.par]):
            raise Exception("Please provide parameters from this list: "+', '.join(supported_pars[:-2]))
    # Read number of single-event draws
    if options.se_draws is None:
        options.se_draws = options.draws
    if options.se_sigma_prior is not None:
        options.se_sigma_prior = np.array([float(s) for s in options.se_sigma_prior.split(',')])
    if options.sigma_prior is not None:
        options.sigma_prior = np.array([float(s) for s in options.sigma_prior.split(',')])
    # Cosmology
    if options.cosmology == 'Planck18':
        approx_dVdz = dVdz_approx_planck18
    else:
        approx_dVdz = dVdz_approx_planck15

    # If provided, load injected density
    inj_density = None
    if options.inj_density_file is not None:
        inj_file_name = Path(options.inj_density_file).parts[-1].split('.')[0]
        spec          = importlib.util.spec_from_file_location(inj_file_name, options.inj_density_file)
        inj_module    = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inj_module)
        inj_density   = inj_module.density
    # If provided, load selecton function
    selfunc     = None
    inj_pdf     = None
    n_total_inj = None
    duration    = 1.
    if options.selfunc_file is not None:
        selfunc, inj_pdf, n_total_inj, duration = load_selection_function(options.selfunc_file,
                                                                          par           = options.par,
                                                                          far_threshold = options.far_threshold,
                                                                          snr_threshold = options.snr_threshold,
                                                                          )
        if not callable(selfunc):
            # Keeping only the samples within bounds
            inj_pdf = inj_pdf[np.where((np.prod(options.bounds[:,0] < selfunc, axis = 1) & np.prod(selfunc < options.bounds[:,1], axis = 1)))]
            selfunc = selfunc[np.where((np.prod(options.bounds[:,0] < selfunc, axis = 1) & np.prod(selfunc < options.bounds[:,1], axis = 1)))]
    if options.include_dvdz and not callable(selfunc):
        raise Exception("The inclusion of dV/dz*(1+z)^{-1} is available only with a selection function approximant.")
    if options.include_dvdz:
        if options.par is None:
            print("Redshift is assumed to be the last parameter")
            z_index = -1
        elif 'z' in options.par:
            z_index = np.where(np.array(options.par)=='z')[0][0]
        else:
            raise Exception("Redshift must be included in the rate analysis")
        dec_selfunc = _decorator_dVdz(selfunc, approx_dVdz, z_index, options.bounds[z_index][1])
    else:
        dec_selfunc = selfunc
    # If provided, load true values
    hier_samples = None
    if options.hier_samples is not None:
        options.hier_samples = Path(options.hier_samples).resolve()
        hier_samples, true_name = load_single_event(options.hier_samples,
                                                    par       = options.par,
                                                    cosmology = options.cosmology,
                                                    waveform  = options.wf,
                                                    )
        if np.shape(hier_samples)[-1] == 1:
            hier_samples = hier_samples.flatten()
    # Load samples
    events, names = load_data(options.input,
                              par        = options.par,
                              n_samples  = options.n_samples_dsp,
                              cosmology  = options.cosmology,
                              waveform   = options.wf,
                              likelihood = options.likelihood,
                              )
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
                                        scale            = options.fraction,
                                        events           = events,
                                        label            = symbols,
                                        unit             = units,
                                        save_se          = options.save_single_event,
                                        MC_draws         = options.mc_draws,
                                        probit           = options.probit,
                                        selfunc          = dec_selfunc,
                                        inj_pdf          = inj_pdf,
                                        n_total_inj      = n_total_inj,
                                        )
                          for _ in range(options.n_parallel)])
        
        if options.run_events:
            # Run each single-event analysis
            posteriors = []
            for s in tqdm(pool.map_unordered(lambda a, v: a.run_event.remote(v), [[ev, name, options.se_draws] for ev, name in zip(events, names)]), total = len(events), desc = 'Events'):
                posteriors.append(s)
            # Save all single-event draws together
            posteriors = np.array(posteriors)
            save_density(posteriors, folder = output_draws, name = 'posteriors_single_event', ext = options.ext)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                posteriors = load_density(Path(output_draws, 'posteriors_single_event.'+options.ext), make_comp = False)
        # Load posteriors
        for s in pool.map(lambda a, v: a.load_posteriors.remote(v), [posteriors for _ in range(options.n_parallel)]):
            pass
        # Run hierarchical analysis
        draws = []
        for s in tqdm(pool.map_unordered(lambda a, v: a.draw_hierarchical.remote(), [_ for _ in range(options.draws)]), total = options.draws, desc = 'Sampling'):
            draws.append(s)
        draws = np.array(draws)
        if options.include_dvdz:
            normalise_alpha_factor(draws, dvdz = approx_dVdz, z_index = z_index, z_max = options.bounds[z_index][1])
        # Save draws
        save_density(draws, folder = output_draws, name = 'draws_'+hier_name, ext = options.ext)
    else:
        draws = load_density(Path(output_draws, 'draws_'+hier_name+'.'+options.ext))
    # Plot
    if dim == 1:
        plot_median_cr(draws,
                       injected     = inj_density,
                       samples      = hier_samples,
                       out_folder   = output_plots,
                       name         = options.hier_name,
                       label        = options.symbol,
                       unit         = options.unit,
                       hierarchical = True,
                       )
    else:
        plot_multidim(draws,
                      samples      = hier_samples,
                      out_folder   = output_plots,
                      name         = hier_name,
                      labels       = symbols,
                      units        = units,
                      hierarchical = True,
                      )

    if options.rate:
        R_samples = sample_rate(draws,
                                n_obs   = len(events),
                                selfunc = selfunc,
                                T       = duration,
                                size    = 1e4,
                                dvdz    = approx_dVdz,
                                z_index = z_index,
                                )
        plot_integrated_rate(R_samples,
                             out_folder = output_rate,
                             name       = options.hier_name,
                             )
        np.savetxt(Path(output_rate, 'samples_integrated_rate_{}.txt'.format(options.hier_name)), R_samples)
        # Best estimate for rate
        rates = sample_rate(draws,
                            n_obs   = len(events),
                            selfunc = selfunc,
                            T       = duration,
                            size    = 1e4,
                            dvdz    = approx_dVdz,
                            z_index = z_index,
                            each    = True,
                            )
        if options.par is not None:
            names = options.par
        else:
            names = np.arange(dim)
        # Marginal rates
        if dim == 1:
            plot_differential_rate(draws,
                                   rate         = rates,
                                   out_folder   = output_rate,
                                   name         = options.hier_name,
                                   label        = options.symbol,
                                   unit         = options.unit,
                                   hierarchical = True,
                                   )
        else:
            for i in range(dim):
                dims = list(np.arange(dim))
                dims.remove(i)
                dd   = marginalise(draws, dims)
                plot_differential_rate(dd,
                                       rate         = rates,
                                       out_folder   = output_rate,
                                       name         = names[i],
                                       label        = symbols[i],
                                       unit         = units[i],
                                       hierarchical = True,
                                       )
if __name__ == '__main__':
    main()
