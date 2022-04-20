import numpy as np

import optparse as op
import configparser
import json
import dill
import importlib

from pathlib import Path
from tqdm import tqdm

from figaro.mixture import DPGMM
from figaro.utils import save_options, plot_median_cr, plot_multidim
from figaro.load import load_single_event

def main():

    parser = op.OptionParser()
    # Input/output
    parser.add_option("-i", "--input", type = "string", dest = "samples_file", help = "File with samples")
    parser.add_option("-b", "--bounds", type = "string", dest = "bounds", help = "Density bounds. Must be a string formatted as '[[xmin, xmax], [ymin, ymax],...]'. For 1D distributions use '[[xmin, xmax]]'", default = None)
    parser.add_option("-o", "--output", type = "string", dest = "output", help = "Output folder. Default: same directory as samples", default = None)
    parser.add_option("--inj_density", type = "string", dest = "inj_density_file", help = "Python module with injected density - please name the method 'density'", default = None)
    parser.add_option("--parameter", type = "string", dest = "par", help = "GW parameter(s) to be read from file", default = 'm1')
    # Plot
    parser.add_option("-p", "--postprocess", dest = "postprocess", action = 'store_true', help = "Postprocessing", default = False)
    parser.add_option("--symbol", type = "string", dest = "symbol", help = "LaTeX-style quantity symbol, for plotting purposes", default = None)
    parser.add_option("--unit", type = "string", dest = "unit", help = "LaTeX-style quantity unit, for plotting purposes", default = None)
    # Settings
    parser.add_option("--draws", type = "int", dest = "n_draws", help = "Number of draws", default = 100)
    parser.add_option("--n_samples_dsp", type = "int", dest = "n_samples_dsp", help = "Number of samples to analyse (downsampling). Default: all", default = -1)
    parser.add_option("--cosmology", type = "string", dest = "cosmology", help = "Cosmological parameters (h, om, ol). Default values from Planck (2021)", default = '0.674,0.315,0.685')

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
        options.bounds = json.loads(options.bounds)
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
    # Read cosmology
    options.h, options.om, options.ol = (float(x) for x in options.cosmology.split(','))
    # Read parameter(s)
    options.par = options.par.split(',')

    save_options(options)
    
    # Load samples
    samples, name = load_single_event(options.samples_file, par = options.par, n_samples = options.n_samples_dsp, h = options.h, om = options.om, ol = options.ol)
    dim = len(samples[0])
    
    # Reconstruction
    if not options.postprocess:
        mix = DPGMM(options.bounds)
        draws = []
        
        for _ in tqdm(range(options.n_draws), desc = name):
            np.random.shuffle(samples)
            mix.density_from_samples(samples)
            draws.append(mix.build_mixture())
            mix.initialise()
            
        draws = np.array(draws)
        with open(Path(options.output, 'draws_'+name+'.pkl'), 'wb') as f:
            dill.dump(draws, f)
    
    else:
        try:
            with open(Path(options.output, 'draws_'+name+'.pkl'), 'rb') as f:
                draws = dill.load(f)
        except FileNotFoundError:
            print("No draws_{0}.pkl file found. Please provide it or re-run the inference".format(name))
            exit()

    # Plot
    if dim == 1:
        plot_median_cr(draws, injected = inj_density, samples = samples, out_folder = options.output, name = name, label = options.symbol, unit = options.unit)
    else:
        if options.symbol is not None:
            symbols = options.symbol.split(',')
        else:
            symbols = options.symbol
        if options.unit is not None:
            units = options.unit.split(',')
        else:
            units = options.unit
        plot_multidim(draws, dim, samples = samples, out_folder = options.output, name = name, labels = symbols, units = units)

if __name__ == '__main__':
    main()
