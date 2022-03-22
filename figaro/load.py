import numpy as np
import os
import h5py
from figaro.cosmology import CosmologicalParameters
from pathlib import Path
from scipy.optimize import newton

def find_redshift(omega, dl):
    def objective(z, omega, dl):
        return dl - omega.LuminosityDistance_double(z)
    return newton(objective,1.0,args=(omega,dl))

def load_single_event(event, seed = 0, par = 'm1', n_samples = -1, h = 0.674, om = 0.315, ol = 0.685):
    '''
    Loads the data from .txt files (for simulations) or .h5/.hdf5 files (posteriors from GWTC) for a single event.
    Default cosmological parameters from Planck Collaboration (2021) in a flat Universe (https://www.aanda.org/articles/aa/pdf/2020/09/aa33910-18.pdf)
    
    Arguments:
        :str file:      file with samples
        :bool seed:     fixes the seed to a default value (1) for reproducibility
        :str par:       parameter to extract from GW posteriors (m1, m2, mc, z, chi_effective)
        :int n_samples: number of samples for (random) downsampling. Default -1: all samples
        :double h:      Hubble constant H0/100 [km/(s*Mpc)]
        :double om:     matter density parameter
        :double ol:     cosmological constant density parameter
    
    Returns:
        :np.ndarray:    samples
        :np.ndarray:    name
    '''
    if not seed == 0:
        rdstate = np.random.RandomState(seed = 1)
    else:
        rdstate = np.random.RandomState()
    name, ext = str(event).split('/')[-1].split('.')
    if ext == 'txt':
        if n_samples > -1:
            samples = np.genfromtxt(event)
            s = int(min([n_samples, len(samples)]))
            out = np.sort(rdstate.choice(samples, size = s, replace = False))
        else:
            out = np.sort(np.genfromtxt(event))
    else:
        out = np.sort(unpack_gw_posterior(event, par = par, n_samples = n_samples, cosmology = (h, om, ol), rdstate = rdstate, ext = ext))
    return out, name

def load_data(path, seed = 0, par = 'm1', n_samples = -1, h = 0.674, om = 0.315, ol = 0.685):
    '''
    Loads the data from .txt files (for simulations) or .h5/.hdf5 files (posteriors from GWTC).
    Default cosmological parameters from Planck Collaboration (2021) in a flat Universe (https://www.aanda.org/articles/aa/pdf/2020/09/aa33910-18.pdf)
    
    Arguments:
        :str path:      folder with data files
        :bool seed:     fixes the seed to a default value (1) for reproducibility
        :str par:       parameter to extract from GW posteriors (m1, m2, mc, z, chi_effective)
        :int n_samples: number of samples for (random) downsampling. Default -1: all samples
        :double h:      Hubble constant H0/100 [km/(s*Mpc)]
        :double om:     matter density parameter
        :double ol:     cosmological constant density parameter
    
    Returns:
        :np.ndarray:    samples
        :np.ndarray:    names
    '''
    if not seed == 0:
        rdstate = np.random.RandomState(seed = 1)
    else:
        rdstate = np.random.RandomState()
        
    event_files = [Path(path,f) for f in os.listdir(path) if not (f.startswith('.') or f.startswith('empty_files'))]
    events      = []
    names       = []
    n_events    = len(event_files)
    
    empty_file_counter = 0
    empty_files        = []
    
    for i, event in enumerate(event_files):
        print('\r{0}/{1} event(s)'.format(i+1, n_events), end = '')
        name, ext = str(event).split('/')[-1].split('.')
        names.append(name)

        
        if ext == 'txt':
            if n_samples > -1:
                if not os.stat(event).st_size == 0:
                    samples = np.atleast_1d(np.genfromtxt(event))
                    s = int(min([n_samples, len(samples)]))
                    events.append(np.sort(rdstate.choice(samples, size = s, replace = False)))
                else:
                    empty_file_counter += 1
                    empty_files.append(str(event))
                    
            else:
                if not os.stat(event).st_size == 0:
                    samples = np.atleast_1d(np.genfromtxt(event))
                    events.append(np.sort(samples))
                else:
                    empty_file_counter += 1
                    empty_files.append(str(event))
                
        else:
            events.append(np.sort(unpack_gw_posterior(event, par = par, n_samples = n_samples, cosmology = (h, om, ol), rdstate = rdstate, ext = ext)))
        
    if empty_file_counter > 0:
        print('\nWarning: {0} empty files detected: these events will not be processed.\nSee empty_files.txt for details.'.format(empty_file_counter))
        np.savetxt(Path(path, 'empty_files.txt'), empty_files, fmt="%s")

    return (events, np.array(names))

def unpack_gw_posterior(event, par, cosmology, rdstate, ext, n_samples = -1):
    '''
    Reads data from .h5/.hdf5 GW posterior files.
    Implemented 'm1', 'm2', 'mc', 'z', 'chi_eff'.
    
    Arguments:
        :str event:       file to read
        :str par:         parameter to extract
        :tuple cosmology: cosmological parameters (h, om, ol)
        :int n_samples:   number of samples for (random) downsampling. Default -1: all samples
    
    Returns:
        :np.ndarray:    samples
    '''
    h, om, ol = cosmology
    omega = CosmologicalParameters(h, om, ol, -1, 0)
    if ext == 'h5' or ext == 'hdf5':
        with h5py.File(Path(event), 'r') as f:
            try:
                data = f['PublicationSamples']['posterior_samples']
                if par == 'm1':
                    samples = data['mass_1_source']
                if par == 'm2':
                    samples = data['mass_2_source']
                if par == 'mc':
                    samples = data['chirp_mass']
                if par == 'z':
                    samples = data['redshift']
                if par == 'chi_eff':
                    samples = data['chi_eff']
                if n_samples > -1:
                    s = int(min([n_samples, len(samples)]))
                    return rdstate.choice(samples, size = s, replace = False)
                else:
                    return samples
            except:
                data = f['Overall_posterior']
                LD        = data['luminosity_distance_Mpc']
                z         = np.array([find_redshift(omega, l) for l in LD])
                m1_detect = data['m1_detector_frame_Msun']
                m2_detect = data['m2_detector_frame_Msun']
                m1        = m1_detect/(1+z)
                m2        = m2_detect/(1+z)
                
                if par == 'z':
                    samples = z
                if par == 'm1':
                    samples = m1
                if par == 'm2':
                    samples = m2
                if par == 'mc':
                    samples = (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
                if par == 'chi_eff':
                    s1   = data['spin1']
                    s2   = data['spin2']
                    cos1 = data['costilt1']
                    cos2 = data['costilt2']
                    q    = m2/m1
                    samples = (s1*cos1 + q*s2*cos2)/(1+q)
                
                if n_samples > -1:
                    s = int(min([n_samples, len(samples)]))
                    return rdstate.choice(samples, size = s, replace = False)
                else:
                    return samples
    else:
        data = np.genfromtxt(Path(event), names = True)
        if par == 'm1':
            samples = data['mass_1']
        if par == 'm2':
            samples = data['mass_2']
        
        if n_samples > -1:
            s = int(min([n_samples, len(samples)]))
            return rdstate.choice(samples, size = s, replace = False)
        else:
            return samples
