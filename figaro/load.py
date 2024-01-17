import numpy as np
import h5py
import warnings
import json
import dill
from figaro.exceptions import FIGAROException
from figaro.mixture import mixture
from figaro.cosmology import CosmologicalParameters
from pathlib import Path
from tqdm import tqdm

supported_extensions = ['h5', 'hdf5', 'txt', 'dat', 'csv']
supported_waveforms  = ['combined', 'imr', 'seob']

GW_par = {'m1'                 : 'mass_1_source',
          'm2'                 : 'mass_2_source',
          'm1_detect'          : 'mass_1',
          'm2_detect'          : 'mass_2',
          'mc'                 : 'chirp_mass_source',
          'mc_detect'          : 'chirp_mass',
          'mt'                 : 'total_mass_source',
          'z'                  : 'redshift',
          'q'                  : 'mass_ratio',
          'eta'                : 'symmetric_mass_ratio',
          'chi_eff'            : 'chi_eff',
          'ra'                 : 'ra',
          'dec'                : 'dec',
          'luminosity_distance': 'luminosity_distance',
          'cos_theta_jn'       : 'cos_theta_jn',
          'cos_tilt_1'         : 'cos_tilt_1',
          'cos_tilt_2'         : 'cos_tilt_2',
          's1x'                : 'spin_1x',
          's2x'                : 'spin_2x',
          's1y'                : 'spin_1y',
          's2y'                : 'spin_2y',
          's1z'                : 'spin_1z',
          's2z'                : 'spin_2z',
          's1'                 : 'spin_1',
          's2'                 : 'spin_2',
          'psi'                : 'psi',
          'cos_iota'           : 'cos_iota',
          'phase'              : 'phase',
          'tc'                 : 'geocent_time',
          'snr'                : 'network_matched_filter_snr',
          'far'                : 'far',
          'log_prior'          : 'log_prior',
          'log_likelihood'     : 'log_likelihood',
          }
    
def available_gw_pars():
    """
    Print a list of available GW parameters
    """
    print([p for p in GW_par.keys() if not p in ['snr', 'far']])

def load_single_event(event, seed = False, par = None, n_samples = -1, h = 0.674, om = 0.315, ol = 0.685, volume = False, waveform = 'combined', snr_threshold = None, far_threshold = None):
    '''
    Loads the data from .txt/.dat files (for simulations) or .h5/.hdf5 files (posteriors from GWTC) for a single event.
    Default cosmological parameters from Planck Collaboration (2021) in a flat Universe (https://www.aanda.org/articles/aa/pdf/2020/09/aa33910-18.pdf)
    Not all GW parameters are implemented: run figaro.load.available_gw_pars() for a list of the available parameters.
    
    Arguments:
        str or Path file: file with samples
        bool seed:        fixes the seed to a default value (1) for reproducibility
        list-of-str par:  list with parameter(s) to extract from GW posteriors (m1, m2, mc, z, chi_effective)
        int n_samples:    number of samples for (random) downsampling. Default -1: all samples
        double h:         Hubble constant H0/100 [km/(s*Mpc)]
        double om:        matter density parameter
        double ol:        cosmological constant density parameter
        bool volume:      if True, loads RA, dec and Luminosity distance (for skymaps)
        str waveform:     waveform family to be used ('combined', 'seob', 'imr')
        double snr_threhsold: SNR threshold for event filtering. For injection analysis only.
        double far_threshold: FAR threshold for event filtering. For injection analysis only.
        
    Returns:
        np.ndarray: samples
        np.ndarray: name
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
    if ext == 'txt' or ext == 'dat' or ext == 'csv':
        if par is not None:
            warnings.warn("Par names (or volume keyword) are ignored for .txt/.dat/.csv files")
        if n_samples > -1:
            samples = np.atleast_1d(np.loadtxt(event))
            s = int(min([n_samples, len(samples)]))
            out = samples[rdstate.choice(np.arange(len(samples)), size = s, replace = False)]
        else:
            out = np.loadtxt(event)
    else:
        # Check that a list of parameters is passed
        if par is None:
            raise TypeError("Please provide a list of parameter names you want to load (e.g. ['m1']).")
        # Check that all the parametes are loadable
        if not np.all([p in GW_par.keys() for p in par]):
            wrong_pars = [p for p in par if not p in GW_par.keys()]
            raise FIGAROException("The following parameters are not implemented: "+", ".join(wrong_pars)+". Run figaro.load.available_gw_pars() for a list of available parameters.")
        # If everything is ok, load the samples
        else:
            out = _unpack_gw_posterior(event, par = par, n_samples = n_samples, cosmology = (h, om, ol), rdstate = rdstate, waveform = waveform, snr_threshold = snr_threshold, far_threshold = far_threshold)
    
    if out is None:
        return out, name
    
    if len(np.shape(out)) == 1:
        out = np.atleast_2d(out).T
    return out, name

def load_data(path, seed = False, par = None, n_samples = -1, h = 0.674, om = 0.315, ol = 0.685, volume = False, waveform = 'combined', snr_threshold = None, far_threshold = None, verbose = True):
    '''
    Loads the data from .txt files (for simulations) or .h5/.hdf5/.dat files (posteriors from GWTC-x).
    Default cosmological parameters from Planck Collaboration (2021) in a flat Universe (https://www.aanda.org/articles/aa/pdf/2020/09/aa33910-18.pdf)
    Not all GW parameters are implemented: run figaro.load.available_gw_pars() for a list of available parameters.
    
    Arguments:
        str or Path path:     folder with data files
        bool seed:            fixes the seed to a default value (1) for reproducibility
        list-of-str par:      list with parameter(s) to extract from GW posteriors
        int n_samples:        number of samples for (random) downsampling. Default -1: all samples
        double h:             Hubble constant H0/100 [km/(s*Mpc)]
        double om:            matter density parameter
        double ol:            cosmological constant density parameter
        str waveform:         waveform family to be used ('combined', 'seob', 'imr')
        double snr_threhsold: SNR threshold for event filtering. For injection analysis only.
        double far_threshold: FAR threshold for event filtering. For injection analysis only.
        bool verbose:         show progress bar

    Returns:
        np.ndarray: samples
        np.ndarray: names
    '''
    folder      = Path(path).resolve()
    event_files = [Path(folder,f) for f in folder.glob('[!.]*')] # Ignores hidden files
    events      = []
    names       = []
    n_events    = len(event_files)
    removed_snr = False
    if volume:
        par = ['ra', 'dec', 'luminosity_distance']
    if n_events == 0:
        raise FIGAROException("Empty folder")
    for event in tqdm(event_files, desc = 'Loading events', disable = not(verbose)):
        if seed:
            rdstate = np.random.RandomState(seed = 1)
        else:
            rdstate = np.random.RandomState()
        name, ext = str(event).split('/')[-1].split('.')
        names.append(name)
        if not ext in supported_extensions:
            raise TypeError("File {0}.{1} is not supported".format(name, ext))
        
        if ext == 'txt' or ext == 'dat' or ext == 'csv':
            if par is not None:
                warnings.warn("Par names (or volume keyword) are ignored for .txt/.dat files")
            if n_samples > -1:
                samples = np.atleast_1d(np.loadtxt(event))
                s = int(min([n_samples, len(samples)]))
                samples_subset = samples[rdstate.choice(np.arange(len(samples)), size = s, replace = False)]
                if len(np.shape(samples_subset)) == 1:
                    samples_subset = np.atleast_2d(samples_subset).T
                events.append(samples_subset)
                    
            else:
                samples = np.atleast_1d(np.loadtxt(event))
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
            # If everything is ok, load the samples
            else:
                out = _unpack_gw_posterior(event, par = par, n_samples = n_samples, cosmology = (h, om, ol), rdstate = rdstate, waveform = waveform, snr_threshold = snr_threshold, far_threshold = far_threshold)
                if out is not None:
                    if out.shape[-1] == len(par):
                        events.append(out)
                    elif 'snr' in par:
                        removed_snr = True
                        names.remove(name)
                else:
                    names.remove(name)
    if removed_snr:
        warnings.warn("At least one event does not have SNR samples. These events cannot be loaded for this parameter choices.")
    return (events, np.array(names))

def _unpack_gw_posterior(event, par, cosmology, rdstate, n_samples = -1, waveform = 'combined', snr_threshold = None, far_threshold = None):
    '''
    Reads data from .h5/.hdf5 GW posterior files.
    For GWTC-3 data release, it uses by default the Mixed posterior samples.
    Not all parameters are implemented: run figaro.load.available_gw_pars() for a list of available parameters.
    The waveform argument allows the user to select a waveform family. The default value, 'combined' uses samples from both imr and seob waveforms.
    For SEOB waveforms, the following waveform models are used (in descending priority order):
        * SEOBNRv4PHM
        * SEOBNRv4P
        * SEOBNRv4
    For IMR waveforms, in descending order:
        * IMRPhenomXPHM
        * IMRPhenomPv2
        * IMRPhenomPv3HM
    
    Arguments:
        str event:            file to read
        str par:              parameter to extract
        tuple cosmology:      cosmological parameters (h, om, ol)
        int n_samples:        number of samples for (random) downsampling. Default -1: all samples
        str waveform:         waveform family to be used ('combined', 'imr', 'seob')
        double snr_threhsold: SNR threshold for event filtering. For injection analysis only.
        double far_threshold: FAR threshold for event filtering. For injection analysis only.
    
    Returns:
        np.ndarray:    samples
    '''
    h, om, ol = cosmology
    omega = CosmologicalParameters(h, om, ol, -1, 0, 0)
    if not waveform in supported_waveforms:
        raise FIGAROException("Unknown waveform: please use 'combined' (default), 'imr' or 'seob'")
    
    if far_threshold is not None and snr_threshold is not None:
        warnings.warn("Both FAR and SNR threshold provided. FAR will be used.")
        snr_threshold = None
    
    if far_threshold is not None:
        if not 'far' in par:
            par = np.append(par, 'far')
    elif snr_threshold is not None:
        if not 'snr' in par:
            par = np.append(par, 'snr')
    
    with h5py.File(Path(event), 'r') as f:
        samples     = []
        loaded_pars = []
        flag_filter = False
        try:
            try:
                # LVK R&P mock data challenge
                try:
                    data = f['MDC']['posterior_samples']
                # Playground
                except:
                    data = f['posterior_samples']
                MDC_flag = True
            # GWTC-2, GWTC-3
            except:
                MDC_flag = False
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
                            try:
                                data = f['C01:IMRPhenomXPHM']['posterior_samples']
                            except:
                                data = f['IMRPhenomXPHM']['posterior_samples']
                        except:
                            try:
                                try:
                                    data = f['C01:IMRPhenomPv2']['posterior_samples']
                                except:
                                    data = f['IMRPhenomPv2']['posterior_samples']
                            except:
                                try:
                                    try:
                                        data = f['C01:IMRPhenomPv3HM']['posterior_samples']
                                    except:
                                        data = f['IMRPhenomPv3HM']['posterior_samples']
                                except:
                                    try:
                                        try:
                                            data = f['C01:IMRPhenomXPHM:LowSpin']['posterior_samples']
                                        except:
                                            data = f['IMRPhenomXPHM:LowSpin']['posterior_samples']
                                    except:
                                        try:
                                            data = f['C01:IMRPhenomPv2_NRTidal-LS']['posterior_samples']
                                        except:
                                            data = f['IMRPhenomPv2_NRTidal-LS']['posterior_samples']
                    if waveform == 'seob':
                        try:
                            try:
                                data = f['C01:SEOBNRv4PHM']['posterior_samples']
                            except:
                                data = f['SEOBNRv4PHM']['posterior_samples']
                        except:
                            try:
                                try:
                                    data = f['C01:SEOBNRv4P']['posterior_samples']
                                except:
                                    data = f['SEOBNRv4P']['posterior_samples']
                            except:
                                try:
                                    try:
                                        data = f['C01:SEOBNRv4']['posterior_samples']
                                    except:
                                        data = f['SEOBNRv4']['posterior_samples']
                                except:
                                    try:
                                        data = f['C01:SEOBNRv4T_surrogate_LS']['posterior_samples']
                                    except:
                                        data = f['SEOBNRv4T_surrogate_LS']['posterior_samples']
                
            for name, lab in zip(GW_par.keys(), GW_par.values()):
                if name in par:
                    if name == 'snr':
                        try:
                            if MDC_flag or waveform != 'combined':
                                snr = np.array(data[lab])
                                samples.append(data[lab])
                            if snr_threshold is not None:
                                flag_filter = True
                        except:
                            warnings.warn("SNR filter is not available with this dataset.")
                    elif name == 'far' and far_threshold is not None:
                        try:
                            flag_filter = True
                            far = np.array(data[lab])
                        except:
                            warnings.warn("FAR filter is not available with this dataset.")
                    elif name == 's1':
                        try:
                            samples.append(data[lab])
                        except:
                            samples.append(np.sqrt(data['spin_1x']**2+data['spin_1y']**2+data['spin_1z']**2))
                    elif name == 's2':
                        try:
                            samples.append(data[lab])
                        except:
                            samples.append(np.sqrt(data['spin_2x']**2+data['spin_2y']**2+data['spin_2z']**2))
                    elif name == 'luminosity_distance':
                        try:
                            samples.append(data[lab])
                        except:
                            samples.append(np.exp(data['logdistance']))
                    else:
                        samples.append(data[lab])
                    loaded_pars.append(name)
            
            if len(par) == 1:
                samples = np.atleast_2d(samples).T
            else:
                par = np.array(par)
                loaded_pars = np.array(loaded_pars)
                samples_loaded = np.array(samples)
                samples = []
                for pi in par:
                    if not (pi == 'far' or (pi == 'snr' and flag_filter)):
                        samples.append(samples_loaded[np.where(loaded_pars == pi)[0]].flatten())
                samples = np.array(samples)
                if flag_filter:
                    if far_threshold is not None:
                        samples = samples[:, np.where((far < far_threshold) & (far > 0))[0]]
                    if snr_threshold is not None:
                        samples = samples[:, np.where(snr > snr_threshold)[0]]
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
            ss['z']       = omega.Redshift(ss['luminosity_distance'])
            ss['m1']      = ss['m1_detect']/(1+ss['z'])
            ss['m2']      = ss['m2_detect']/(1+ss['z'])
            ss['mc']      = (ss['m1']*ss['m2'])**(3./5.)/(ss['m1']+ss['m2'])**(1./5.)
            ss['mt']      = ss['m1']+ss['m2']
            ss['q']       = ss['m2']/ss['m1']
            ss['eta']     = ss['m1']*ss['m2']/(ss['m1']+ss['m2'])**2
            ss['s1z']     = ss['s1']*ss['cos_tilt_1']
            ss['s2z']     = ss['s2']*ss['cos_tilt_2']
            ss['chi_eff'] = (ss['s1z'] + ss['q']*ss['s2z'])/(1+ss['q'])

            for name in GW_par.keys():
                if name in par:
                    if not (name == 'snr' or name == 'far'):
                        samples.append(ss[name])
                        loaded_pars.append(name)

            if len(par) == 1:
                samples = np.atleast_2d(samples).T
            else:
                par = np.array(par)
                loaded_pars = np.array(loaded_pars)
                samples_loaded = np.array(samples)
                samples = []
                for pi in par:
                    if not (pi == 'snr' or pi == 'far'):
                        samples.append(samples_loaded[np.where(loaded_pars == pi)[0]].flatten())
                samples = np.array(samples)
                samples = samples.T
            if n_samples > -1:
                s = int(min([n_samples, len(samples)]))
                return samples[rdstate.choice(np.arange(len(samples)), size = s, replace = False)]
            else:
                return samples

def save_density(draws, folder = '.', name = 'density', ext = 'pkl'):
    """
    Exports a list of figaro.mixture instances to file

    Arguments:
        list draws:         list of mixtures to be saved
        str or Path folder: folder in which the output file will be saved
        str name:           name to be given to output file
        str ext:            file extension (pkl or json)
    """
    if ext == 'pkl':
        with open(Path(folder, name+'.pkl'), 'wb') as f:
            dill.dump(draws, f)
    elif ext == 'json':
        if len(np.shape(draws)) == 1:
            draws = np.atleast_2d(draws)
        ll = []
        for draws_i in draws:
            list_of_dicts = [dr.__dict__.copy() for dr in draws_i]
            for density in list_of_dicts:
                for key in density.keys():
                    value = density[key]
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    density[key] = value
            ll.append(list_of_dicts)
        s = json.dumps(ll)
        with open(Path(folder, name + '.json'), 'w') as f:
            json.dump(s, f)
    else:
        raise FIGAROException("Extension {0} is not supported. Valid extensions are pkl or json.")

def load_density(path):
    """
    Loads a list of figaro.mixture instances from path.
    If the requested file extension (pkl or json) is not available, it tries loading the other.

    Arguments:
        str or Path path: path with draws (file or folder)

    Returns:
        list: figaro.mixture object instances
    """
    path = Path(path)
    if path.is_file():
        return _load_density_file(path)
    else:
        files = [_load_density_file(file) for file in path.glob('*.[jp][sk][ol]*') if not file.stem == 'posteriors_single_event']
        if len(files) > 0:
            return files
        else:
            raise FIGAROException("Density file(s) not found")

def _load_density_file(file):
    """
    Loads a list of figaro.mixture instances from file.
    If the requested file extension (pkl or json) is not available, it tries loading the other.

    Arguments:
        str or Path file: file with draws

    Returns
        list: figaro.mixture object instances
    """
    file = Path(file)
    ext  = file.suffix
    if ext == '.pkl':
        try:
            return _load_pkl(file)
        except FileNotFoundError:
            try:
                return _load_json(file.with_suffix('.json'))
            except FileNotFoundError:
                raise FIGAROException("{0} not found. Please provide it or re-run the inference.".format(file.name))
    elif ext == '.json':
        try:
            return _load_json(file)
        except FileNotFoundError:
            try:
                return _load_pkl(file.with_suffix('.pkl'))
            except FileNotFoundError:
                raise FIGAROException("{0} not found. Please provide it or re-run the inference.".format(file.name))
    else:
        raise FIGAROException("Extension {0} is not supported. Please provide .pkl or .json file.".format(file.suffix))

def _load_pkl(file):
    """
    Loads a list of figaro.mixture instances from pkl file

    Arguments:
        str or Path file: file with draws

    Returns
        list: figaro.mixture object instances
    """
    with open(file, 'rb') as f:
        draws = dill.load(f)
    return draws

def _load_json(file):
    """
    Loads a list of figaro.mixture instances from json file

    Arguments:
        str or Path file: file with draws

    Returns
        list: figaro.mixture object instances
    """
    with open(Path(file), 'r') as fjson:
        dictjson = json.loads(json.load(fjson))
    ll = []
    for list_of_dict in dictjson:
        draws = []
        for dict_ in list_of_dict:
            dict_.pop('log_w')
            for key in dict_.keys():
                value = dict_[key]
                if isinstance(value, list):
                    dict_[key] = np.array(value)
            draws.append(mixture(**dict_))
        ll.append(draws)
    if len(ll) == 1:
        return ll[0]
    return ll

