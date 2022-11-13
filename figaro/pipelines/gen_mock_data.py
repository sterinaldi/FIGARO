import numpy as np
import matplotlib.pyplot as plt

from figaro.utils import rejection_sampler

import optparse as op
import importlib

from scipy.stats import norm

from pathlib import Path
import json

def main():
    
    parser = op.OptionParser()
    # Input/output
    parser.add_option("-o", "--output", type = "string", dest = "output", help = "Output folder. Default: working directory", default = '.')
    parser.add_option("-n", "--n_evs", type = "int", dest = "n_evs", help = "Number of events", default = 250)
    parser.add_option("--n_samples", type = "int", dest = "n_samples", help = "Number of samples per event", default = 1000)
    parser.add_option("--pdf", type = "string", dest = "pdf_file", help = "Python module with injected pdf - please name the method 'density'", default = None)
    parser.add_option("--selfunc", type = "string", dest = "selfunc_file", help = "Python module with selection function - please name the method 'selfunc'", default = None)
    parser.add_option("-b", "--bounds", type = "string", dest = "bounds", help = "Minimum and maximum value", default = '-5,5')
    # Plot
    parser.add_option("--symbol", type = "string", dest = "symbol", help = "LaTeX-style quantity symbol, for plotting purposes", default = 'x')
    parser.add_option("--unit", type = "string", dest = "unit", help = "LaTeX-style quantity unit, for plotting purposes", default = None)
    
    (options, args) = parser.parse_args()

    # Paths
    options.output = Path(options.output).resolve()
    options.ev_folder = Path(options.output, 'events').resolve()
    if not options.ev_folder.exists():
        options.ev_folder.mkdir(parents=True)
    
    # PDF
    if options.pdf_file is not None:
        pdf_file_name = Path(options.pdf_file).parts[-1].split('.')[0]
        spec = importlib.util.spec_from_file_location(pdf_file_name, options.pdf_file)
        pdf_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pdf_module)
        pdf = pdf_module.density
    else:
    	# Default value
        pdf = norm().pdf

    # Selection function
    if options.selfunc_file is not None:
        selfunc_file_name = Path(options.selfunc_file).parts[-1].split('.')[0]
        spec = importlib.util.spec_from_file_location(selfunc_file_name, options.selfunc_file)
        selfunc_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(selfunc_module)
        selfunc = selfunc_module.selection_function
    else:
        selfunc = None
    
    # Bounds
    bounds = [float(b) for b in options.bounds.split(',')]

    # Draw true values
    true_vals = rejection_sampler(options.n_evs, pdf, bounds, selfunc = selfunc)
    # Draw posterior widths
    widths = np.random.uniform(2,4, size = options.n_evs)
    
    true_vals_dict = {}
    all_samples    = []
    for i in range(options.n_evs):
    
        # Store true value
        true_vals_dict[str(i+1)] = true_vals[i]
        
        # Draw samples
        mean_val = np.random.normal(true_vals[i], widths[i], 1)
        samples  = np.random.normal(mean_val, widths[i], options.n_samples)
        all_samples.extend(samples)
        
        # Save samples
        np.savetxt(Path(options.ev_folder,'{0}.txt'.format(i+1)), samples)
    
    #Save true values as JSON file
    with open(Path(options.output, 'true_vals.json'), 'w', encoding='utf-8') as f:
        json.dump(true_vals_dict, f, ensure_ascii=False)
    # Save true values as txt file as well
    np.savetxt(Path(options.output, 'truths.txt'), true_vals)
    
    x  = np.linspace(bounds[0],bounds[1],1000)
    dx = x[1]-x[0]
    # Make plot
    plt.hist(all_samples, bins = int(np.sqrt(len(all_samples))), density = True, histtype = 'step', label = 'Samples (stacked)', color = 'forestgreen', alpha = 0.5)
    plt.hist(true_vals, bins = int(np.sqrt(len(true_vals))), density = True, histtype = 'step', label = 'True values', color = 'steelblue')
    plt.plot(x, pdf(x), c = 'r', lw = 0.7, label = 'Injected')
    
    if selfunc is not None:
        filtered = density(x)*selection_function(x)
        plt.plot(x, filtered/np.sum(filtered*dx), c = 'k', lw = 0.7, label = 'After selection effects')
    
    if options.unit is not None:
        plt.xlabel('${0}\ [{1}]$'.format(options.symbol, options.unit))
    else:
        plt.xlabel('${0}$'.format(options.symbol))
    plt.ylabel('$p({0})$'.format(options.symbol))
    plt.grid(alpha = 0.5)
    plt.legend(loc=0, frameon=False)
    plt.savefig(Path(options.output, 'truths.pdf'), bbox_inches = 'tight')

if __name__ == '__main__':
    main()
