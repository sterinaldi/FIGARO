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

def load_single_event(event, seed = 0, par = ['m1'], n_samples = -1, h = 0.674, om = 0.315, ol = 0.685):
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
            out = rdstate.choice(samples, size = s, replace = False)
        else:
            out = np.genfromtxt(event)
    else:
        out = unpack_gw_posterior(event, par = par, n_samples = n_samples, cosmology = (h, om, ol), rdstate = rdstate, ext = ext)
    return out, name

def load_data(path, seed = 0, par = ['m1'], n_samples = -1, h = 0.674, om = 0.315, ol = 0.685):
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
    
    for i, event in enumerate(event_files):
        print('\r{0}/{1} event(s)'.format(i+1, n_events), end = '')
        name, ext = str(event).split('/')[-1].split('.')
        names.append(name)

        
        if ext == 'txt':
            if n_samples > -1:
                samples = np.atleast_1d(np.genfromtxt(event))
                s = int(min([n_samples, len(samples)]))
                events.append(rdstate.choice(samples, size = s, replace = False))
                    
            else:
                samples = np.atleast_1d(np.genfromtxt(event))
                events.append(samples)
                
        else:
            events.append(unpack_gw_posterior(event, par = par, n_samples = n_samples, cosmology = (h, om, ol), rdstate = rdstate, ext = ext))

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
            samples = []
            try:
                data = f['PublicationSamples']['posterior_samples']
                if 'm1' in par:
                    samples.append(data['mass_1_source'])
                if 'm2' in par:
                    samples.append(data['mass_2_source'])
                if 'mc' in par:
                    samples.append(data['chirp_mass'])
                if 'z' in par:
                    samples.append(data['redshift'])
                if 'chi_eff' in par:
                    samples.append(data['chi_eff'])
                if 'ra' in par:
                    samples.append(data['ra'])
                if 'dec' in par:
                    samples.append(data['dec'])
                if 'luminosity_distance' in par:
                    samples.append(data['luminosity_distance'])
                
                if len(par) == 1:
                    samples = np.array(samples[0])
                else:
                    samples = np.array(samples).T

                if n_samples > -1:
                    s = int(min([n_samples, len(samples)]))
                    return rdstate.choice(samples, size = s, replace = False)
                else:
                    return samples
            except:
                data = f['Overall_posterior']
                ra        = data['right_ascension']
                dec       = data['declination']
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
                if 'ra' in par:
                    samples.append(ra)
                if 'dec' in par:
                    samples.append(dec)
                if 'luminosity_distance' in par:
                    samples.append(LD)
                
                if len(par) == 1:
                    samples = np.array(samples[0])
                else:
                    samples = np.array(samples).T

                if n_samples > -1:
                    s = int(min([n_samples, len(samples)]))
                    return rdstate.choice(samples, size = s, replace = False)
                else:
                    return samples
    else:
        data = np.genfromtxt(Path(event), names = True)
        
        samples = []
        
        ra        = data['ra']
        dec       = data['dec']
        LD        = data['luminosity_distance']
        z         = np.array([find_redshift(omega, l) for l in LD])
        m1_detect = data['mass_1']
        m2_detect = data['mass_2']
        m1        = m1_detect/(1+z)
        m2        = m2_detect/(1+z)
        
        if 'z' in par:
            samples.append(z)
        if 'm1' in par:
            samples.append(m1)
        if 'm2' in par:
            samples.append(m2)
        if 'mc' in par:
            samples.append((m1*m2)**(3./5.)/(m1+m2)**(1./5.))
        if 'ra' in par:
            samples.append(ra)
        if 'dec' in par:
            samples.append(dec)
        if 'luminosity_distance' in par:
            samples.append(LD)
        
        if len(par) == 1:
            samples = np.array(samples[0])
        else:
            samples = np.array(samples).T
        
        if n_samples > -1:
            s = int(min([n_samples, len(samples)]))
            return rdstate.choice(samples, size = s, replace = False)
        else:
            return samples
