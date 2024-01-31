import numpy as np

import optparse as op
import dill
import importlib

from pathlib import Path
from tqdm import tqdm
from warnings import warn

from figaro.mixture import DPGMM
from figaro.utils import save_options, load_options, get_priors
from figaro.plot import plot_median_cr, plot_multidim
from figaro.load import load_single_event, save_density, load_density, supported_extensions

def main():

    parser = op.OptionParser()
    # Input/output
    parser.add_option("-i", "--input", type = "string", dest = "samples_path", help = "File with samples", default = None)
    parser.add_option("-b", "--bounds", type = "string", dest = "bounds", help = "Density bounds. Must be a string formatted as '[[xmin, xmax], [ymin, ymax],...]'. For 1D distributions use '[xmin, xmax]'. Quotation marks are required and scientific notation is accepted", default = None)
    parser.add_option("-o", "--output", type = "string", dest = "output", help = "Output folder. Default: same directory as samples", default = None)
    parser.add_option("-j", dest = "json", action = 'store_true', help = "Save mixtures in json file", default = False)
    parser.add_option("--inj_density", type = "string", dest = "inj_density_file", help = "Python module with injected density - please name the method 'density'", default = None)
    parser.add_option("--selfunc", type = "string", dest = "selfunc_file", help = "Python module with selection function - please name the method 'selection_function'", default = None)
    parser.add_option("--parameter", type = "string", dest = "par", help = "GW parameter(s) to be read from file", default = None)
    parser.add_option("--waveform", type = "string", dest = "wf", help = "Waveform to load from samples file. To be used in combination with --parameter. Accepted values: 'combined', 'imr', 'seob'", default = 'combined')
    # Plot
    parser.add_option("-p", "--postprocess", dest = "postprocess", action = 'store_true', help = "Postprocessing", default = False)
    parser.add_option("--symbol", type = "string", dest = "symbol", help = "LaTeX-style quantity symbol, for plotting purposes", default = None)
    parser.add_option("--unit", type = "string", dest = "unit", help = "LaTeX-style quantity unit, for plotting purposes", default = None)
    # Settings
    parser.add_option("--draws", type = "int", dest = "n_draws", help = "Number of draws", default = 100)
    parser.add_option("--n_samples_dsp", type = "int", dest = "n_samples_dsp", help = "Number of samples to analyse (downsampling). Default: all", default = -1)
    parser.add_option("--exclude_points", dest = "exclude_points", action = 'store_true', help = "Exclude points outside bounds from analysis", default = False)
    parser.add_option("--cosmology", type = "string", dest = "cosmology", help = "Cosmological parameters (h, om, ol). Default values from Planck (2021)", default = '0.674,0.315,0.685')
    parser.add_option("--sigma_prior", dest = "sigma_prior", type = "string", help = "Expected standard deviation (prior) - single value or n-dim values. If None, it is estimated from samples", default = None)
    parser.add_option("--fraction", dest = "scale", type = "float", help = "Fraction of samples standard deviation for sigma prior. Overrided by sigma_prior.", default = None)
    parser.add_option("--snr_threshold", dest = "snr_threshold", type = "float", help = "SNR threshold for simulated GW datasets", default = None)
    parser.add_option("--far_threshold", dest = "far_threshold", type = "float", help = "FAR threshold for simulated GW datasets", default = None)
    parser.add_option("--no_probit", dest = "probit", action = 'store_false', help = "Disable probit transformation", default = True)
    parser.add_option("--config", dest = "config", type = "string", help = "Config file. Warning: command line options are ignored if provided", default = None)
    
    (options, args) = parser.parse_args()

    # Paths
    if options.samples_path is not None:
        options.samples_path = Path(options.samples_path).resolve()
    elif options.config is not None:
        options.samples_path = Path('.').resolve()
    else:
        raise Exception("Please provide path to samples.")
    if options.output is not None:
        options.output = Path(options.output).resolve()
        if not options.output.exists():
            options.output.mkdir(parents=True)
    else:
        options.output = options.samples_path.parent
    if options.config is not None:
        options.config = Path(options.config).resolve()
    # File extension
    if options.json:
        options.ext = 'json'
    else:
        options.ext = 'pkl'
    
    if options.config is not None:
        load_options(options, options.config)
    save_options(options, options.output)

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
    
    if options.sigma_prior is not None:
        options.sigma_prior = np.array([float(s) for s in options.sigma_prior.split(',')])
    if options.samples_path.is_file():
        files = [options.samples_path]
        output_draws = options.output
        subfolder = False
    else:
        files = sum([list(options.samples_path.glob('*.'+ext)) for ext in supported_extensions], [])
        output_draws = Path(options.output, 'draws')
        if not output_draws.exists():
            output_draws.mkdir()
        subfolder = True
    if options.exclude_points:
        print("Ignoring points outside bounds.")
    
    for i, file in enumerate(files):
        # Load samples
        samples, name = load_single_event(file, par = options.par, n_samples = options.n_samples_dsp, h = options.h, om = options.om, ol = options.ol, waveform = options.wf, snr_threshold = options.snr_threshold, far_threshold = options.far_threshold)
        try:
            dim = np.shape(samples)[-1]
        except IndexError:
            dim = 1
        if options.exclude_points:
            samples = samples[np.where((np.prod(options.bounds[:,0] < samples, axis = 1) & np.prod(samples < options.bounds[:,1], axis = 1)))]
        else:
            # Check if all samples are within bounds
            if options.probit:
                if not np.all([(samples[:,i] > options.bounds[i,0]).all() and (samples[:,i] < options.bounds[i,1]).all() for i in range(dim)]):
                    raise ValueError("One or more samples are outside the given bounds.")

        # Reconstruction
        if not options.postprocess:
            # Actual analysis
            prior_pars = get_priors(options.bounds, samples = samples, std = options.sigma_prior, scale = options.scale, probit = options.probit, hierarchical = False)
            mix        = DPGMM(options.bounds, prior_pars = prior_pars, probit = options.probit)
            desc       = name + ' ({0}/{1})'.format(i+1, len(files))
            draws      = np.array([mix.density_from_samples(samples) for _ in tqdm(range(options.n_draws), desc = desc)])
            # Save reconstruction
            save_density(draws, folder = output_draws, name = 'draws_'+name, ext = options.ext)
        else:
            draws = load_density(Path(output_draws, 'draws_'+name+'.'+options.ext))

        # Plot
        if dim == 1:
            plot_median_cr(draws, injected = inj_density, selfunc = selfunc, samples = samples, out_folder = options.output, name = name, label = options.symbol, unit = options.unit, subfolder = subfolder)
        else:
            if options.symbol is not None:
                symbols = options.symbol.split(',')
            else:
                symbols = options.symbol
            if options.unit is not None:
                units = options.unit.split(',')
            else:
                units = options.unit
            plot_multidim(draws, samples = samples, out_folder = options.output, name = name, labels = symbols, units = units, subfolder = subfolder)

if __name__ == '__main__':
    main()
