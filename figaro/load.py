import numpy as np
import h5py
import warnings
import json
from figaro.exceptions import FIGAROException
from figaro.mixture import mixture
try:
    from figaro.cosmology import CosmologicalParameters
    lal_flag = True
except ModuleNotFoundError:
    lal_flag = False
from pathlib import Path
from scipy.optimize import newton
from tqdm import tqdm

supported_extensions = ['h5', 'hdf5', 'txt', 'dat']
supported_waveforms  = ['combined', 'imr', 'seob']

GW_par = {'m1'                 : 'mass_1_source',
          'm2'                 : 'mass_2_source',
          'm1_detect'          : 'mass_1',
          'm2_detect'          : 'mass_2',
          'mc'                 : 'chirp_mass_source',
          'mt'                 : 'total_mass_source',
          'z'                  : 'redshift',
          'q'                  : 'mass_ratio',
          'chi_eff'            : 'chi_eff',
          'ra'                 : 'ra',
          'dec'                : 'dec',
          'luminosity_distance': 'luminosity_distance',
          'cos_theta_jn'       : 'cos_theta_jn',
          'cos_tilt_1'         : 'cos_tilt_1',
          'cos_tilt_2'         : 'cos_tilt_2',
          's1z'                : 'spin_1z',
          's2z'                : 'spin_2z',
          's1'                 : 'spin_1',
          's2'                 : 'spin_2',
          }

def _find_redshift(omega, dl):
    """
    Find redshift given a luminosity distance and a cosmology using Newton's method
    
    Arguments:
        :CosmologicalParameters omega: cosmology (see cosmology.pyx for definition)
        :double dl:                    luminosity distance
    
    Returns:
        :double: redshift
    """
    def objective(z, omega, dl):
        return dl - omega.LuminosityDistance_double(z)
    return newton(objective,1.0,args=(omega,dl))
    
def available_gw_pars():
    """
    Print a list of available GW parameters
    """
    print([p for p in GW_par.keys()])

def load_single_event(event, seed = False, par = None, n_samples = -1, h = 0.674, om = 0.315, ol = 0.685, volume = False, waveform = 'combined'):
    '''
    Loads the data from .txt/.dat files (for simulations) or .h5/.hdf5 files (posteriors from GWTC) for a single event.
    Default cosmological parameters from Planck Collaboration (2021) in a flat Universe (https://www.aanda.org/articles/aa/pdf/2020/09/aa33910-18.pdf)
    Not all GW parameters are implemented: run figaro.load.available_gw_pars() for a list of the available parameters.
    
    Arguments:
        :str or Path file: file with samples
        :bool seed:        fixes the seed to a default value (1) for reproducibility
        :list-of-str par:  list with parameter(s) to extract from GW posteriors (m1, m2, mc, z, chi_effective)
        :int n_samples:    number of samples for (random) downsampling. Default -1: all samples
        :double h:         Hubble constant H0/100 [km/(s*Mpc)]
        :double om:        matter density parameter
        :double ol:        cosmological constant density parameter
    
    Returns:
        :np.ndarray: samples
        :np.ndarray: name
    '''
    if seed:
        rdstate = np.random.RandomState(seed = 1)
    else:
        rdstate = np.random.RandomState()
    event = Path(event).resolve()
    name, ext = str(event).split('/')[-1].split('.')
    if volume:
        par = ['ra', 'dec', 'luminosity_distance']
    if not ext in supported_extensions:
        raise TypeError("File {0}.{1} is not supported".format(name, ext))
    if ext == 'txt' or ext == 'dat':
        if par is not None:
            warnings.warn("Par names (or volume keyword) are ignored for .txt/.dat files")
        if n_samples > -1:
            samples = np.atleast_1d(np.genfromtxt(event))
            s = int(min([n_samples, len(samples)]))
            out = samples[rdstate.choice(np.arange(len(samples)), size = s, replace = False)]
        else:
            out = np.genfromtxt(event)
    else:
        # Check that a list of parameters is passed
        if par is None:
            raise TypeError("Please provide a list of parameter names you want to load (e.g. ['m1']).")
        # Check that all the parametes are loadable
        if not np.all([p in GW_par.keys() for p in par]):
            wrong_pars = [p for p in par if not p in GW_par.keys()]
            raise FIGAROException("The following parameters are not implemented: "+", ".join(wrong_pars)+". Run figaro.load.available_gw_pars() for a list of available parameters.")
        # Check if lal is installed
        if not lal_flag:
            raise FIGAROException("LAL is not installed. GW posterior samples cannot be loaded.")
        # If everything is ok, load the samples
        else:
            out = _unpack_gw_posterior(event, par = par, n_samples = n_samples, cosmology = (h, om, ol), rdstate = rdstate, waveform = waveform)
    
    if out is None:
        return out, name
    
    if len(np.shape(out)) == 1:
        out = np.atleast_2d(out).T
    return out, name

def load_data(path, seed = False, par = None, n_samples = -1, h = 0.674, om = 0.315, ol = 0.685, volume = False, waveform = 'combined'):
    '''
    Loads the data from .txt files (for simulations) or .h5/.hdf5/.dat files (posteriors from GWTC-x).
    Default cosmological parameters from Planck Collaboration (2021) in a flat Universe (https://www.aanda.org/articles/aa/pdf/2020/09/aa33910-18.pdf)
    Not all GW parameters are implemented: run figaro.load.available_gw_pars() for a list of available parameters.
    
    Arguments:
        :str or Path path: folder with data files
        :bool seed:        fixes the seed to a default value (1) for reproducibility
        :list-of-str par:  list with parameter(s) to extract from GW posteriors
        :int n_samples:    number of samples for (random) downsampling. Default -1: all samples
        :double h:         Hubble constant H0/100 [km/(s*Mpc)]
        :double om:        matter density parameter
        :double ol:        cosmological constant density parameter
    
    Returns:
        :np.ndarray: samples
        :np.ndarray: names
    '''
    folder      = Path(path).resolve()
    event_files = [Path(folder,f) for f in folder.glob('[!.]*')] # Ignores hidden files
    events      = []
    names       = []
    n_events    = len(event_files)
    if volume:
        par = ['ra', 'dec', 'luminosity_distance']
    for event in tqdm(event_files, desc = 'Loading events'):
        if seed:
            rdstate = np.random.RandomState(seed = 1)
        else:
            rdstate = np.random.RandomState()
        name, ext = str(event).split('/')[-1].split('.')
        names.append(name)
        if not ext in supported_extensions:
            raise TypeError("File {0}.{1} is not supported".format(name, ext))
        
        if ext == 'txt' or ext == 'dat':
            if par is not None:
                warnings.warn("Par names (or volume keyword) are ignored for .txt/.dat files")
            if n_samples > -1:
                samples = np.atleast_1d(np.genfromtxt(event))
                s = int(min([n_samples, len(samples)]))
                samples_subset = samples[rdstate.choice(np.arange(len(samples)), size = s, replace = False)]
                if len(np.shape(samples_subset)) == 1:
                    samples_subset = np.atleast_2d(samples_subset).T
                events.append(samples_subset)
                    
            else:
                samples = np.atleast_1d(np.genfromtxt(event))
                if len(np.shape(samples)) == 1:
                    samples = np.atleast_2d(samples).T
                events.append(samples)
                
        else:
            # Check that a list of parameters is passed
            if par is None:
                raise TypeError("Please provide a list of parameter names you want to load (e.g. ['m1']).")
            # Check that all the parametes are loadable
            if not np.all([p in GW_par.keys() for p in par]):
                wrong_pars = [p for p in par if not p in GW_par.keys()]
                raise FIGAROException("The following parameters are not implemented: "+", ".join(wrong_pars)+". Run figaro.load.available_gw_pars() for a list of available parameters.")
            # Check if lal is installed
            if not lal_flag:
                raise FIGAROException("LAL is not installed. GW posterior samples cannot be loaded.")
            # If everything is ok, load the samples
            else:
                out = _unpack_gw_posterior(event, par = par, n_samples = n_samples, cosmology = (h, om, ol), rdstate = rdstate, waveform = waveform)
                if out is not None:
                    events.append(out)
                else:
                    names.remove(name)
                
    return (events, np.array(names))

def _unpack_gw_posterior(event, par, cosmology, rdstate, n_samples = -1, waveform = 'combined'):
    '''
    Reads data from .h5/.hdf5 GW posterior files.
    For GWTC-3 data release, it uses by default the Mixed posterior samples.
    Not all parameters are implemented: run figaro.load.available_gw_pars() for a list of available parameters.
    
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
    if not waveform in supported_waveforms:
        raise FIGAROException("Unknown waveform: please use 'combined' (default), 'imr' or 'seob'")
        
    with h5py.File(Path(event), 'r') as f:
        samples     = []
        loaded_pars = []
        # GWTC-2, GWTC-3
        try:
            if waveform == 'combined':
                # GWTC-2
                try:
                    data = f['PublicationSamples']['posterior_samples']
                # GWTC-3
                except KeyError:
                    try:
                        data = f['C01:Mixed']['posterior_samples']
                    except:
                        try:
                            data = f['IMRPhenomXPHM']['posterior_samples']
                        except:
                            data = f['SEOBNRv4PHM']['posterior_samples']
            else:
                if waveform == 'imr':
                    try:
                        data = f['C01:IMRPhenomXPHM']['posterior_samples']
                    except:
                        data = f['C01:IMRPhenomPv2']['posterior_samples']
                elif waveform == 'seob':
                    try:
                        data = f['C01:SEOBNRv4PHM']['posterior_samples']
                    except:
                        try:
                            data = f['C01:SEOBNRv4P']['posterior_samples']
                        except:
                            data = f['C01:SEOBNRv4']['posterior_samples']
                
                
            for name, lab in zip(GW_par.keys(), GW_par.values()):
                if name in par:
                    if name == 's1':
                        samples.append(np.sqrt(data['spin_1x']**2+data['spin_1y']**2+data['spin_1z']**2))
                    elif name == 's2':
                        samples.append(np.sqrt(data['spin_2x']**2+data['spin_2y']**2+data['spin_2z']**2))
                    else:
                        samples.append(data[lab])
                    loaded_pars.append(name)
            
            if len(par) == 1:
                samples = np.atleast_2d(samples).T
            else:
                par = np.array(par)
                loaded_pars = np.array(loaded_pars)
                samples = np.array(samples)
                samples = samples[np.array([np.where(pi == par) for pi in loaded_pars]).flatten()]
                samples = samples.T

            if n_samples > -1:
                s = int(min([n_samples, len(samples)]))
                return samples[rdstate.choice(np.arange(len(samples)), size = s, replace = False)]
            else:
                return samples

        # GWTC-1
        except KeyError:
            if waveform == 'combined':
                label = 'Overall_posterior'
            elif waveform == 'imr':
                label = 'IMRPhenomPv2_posterior'
            elif waveform == 'seob':
                label = 'SEOBNRv3_posterior'
            try:
                data = f[label]
            except KeyError:
                try:
                    # GW170817
                    data = f['IMRPhenomPv2NRT_lowSpin_posterior']
                except:
                    print("Skipped event {0} (not loadable yet)".format(Path(event).parts[-1].split('.')[0]))
                    return None
            
            # Provided quantities
            names_GWTC1 = {'m1_detect'          : 'm1_detector_frame_Msun',
                           'm2_detect'          : 'm2_detector_frame_Msun',
                           'ra'                 : 'right_ascension',
                           'dec'                : 'declination',
                           'luminosity_distance': 'luminosity_distance_Mpc',
                           'cos_theta_jn'       : 'costheta_jn',
                           's1'                 : 'spin1',
                           's2'                 : 'spin2',
                           'cos_tilt_1'         : 'costilt1',
                           'cos_tilt_2'         : 'costilt2',
                           }
            
            ss = {name: data[lab] for name, lab in zip(names_GWTC1.keys(), names_GWTC1.values())}

            # Derived quantities
            ss['z']       = np.array([_find_redshift(omega, l) for l in ss['luminosity_distance']])
            ss['m1']      = ss['m1_detect']/(1+ss['z'])
            ss['m2']      = ss['m2_detect']/(1+ss['z'])
            ss['mc']      = (ss['m1']*ss['m2'])**(3./5.)/(ss['m1']+ss['m2'])**(1./5.)
            ss['mt']      = ss['m1']+ss['m2']
            ss['q']       = ss['m2']/ss['m1']
            ss['s1z']     = ss['s1']*ss['cos_tilt_1']
            ss['s2z']     = ss['s2']*ss['cos_tilt_2']
            ss['chi_eff'] = (ss['s1z'] + ss['q']*ss['s2z'])/(1+ss['q'])


            for name in GW_par.keys():
                if name in par:
                    samples.append(ss[name])
                    loaded_pars.append(name)

            if len(par) == 1:
                samples = np.atleast_2d(samples).T
            else:
                par = np.array(par)
                loaded_pars = np.array(loaded_pars)
                samples = np.array(samples)
                samples = samples[np.array([np.where(pi == par) for pi in loaded_pars]).flatten()]
                samples = samples.T

            if n_samples > -1:
                s = int(min([n_samples, len(samples)]))
                return samples[rdstate.choice(np.arange(len(samples)), size = s, replace = False)]
            else:
                return samples

def save_density(density, folder = '.', name = 'density'):
    """
    Exports a figaro.mixture instance into a json file

    Arguments:
        :figaro.mixture density: mixture to be saved for later analysis.
        :string or Path folder:  The folder in which the output json file will be saved.
        :string name:            Name to be given to output file.
    """
    dict_ = density.__dict__.copy()

    for key in dict_.keys():
        value = dict_[key]
        if isinstance(value, np.ndarray):
            value = value.tolist()
        dict_[key] = value
        
    s = json.dumps(dict_, indent=4)

    with open(Path(folder, name + '.json'), 'w') as f:
        json.dump(s, f)

def load_density(file):
    """
    Reads a json file containing the parameters for a saved figaro.mixture object and returns an instance of such object.

    Arguments:
        :string or Path file:  The path to the json file of the mixture.
        
    Returns
        :figaro.mixture: An instance of the class containing the data stored in the json file.
    """

    with open(Path(file), 'r') as fjson:
        dictjson = json.load(fjson)

    dict_ = json.loads(dictjson)
    dict_.pop('log_w')

    for key in dict_.keys():
        value = dict_[key]
        if isinstance(value, list):
            dict_[key] = np.array(value)
    density = mixture(**dict_)

    return density
