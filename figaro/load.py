import numpy as np
import h5py
import warnings
import json
import dill
import copy
import importlib
from figaro.exceptions import FIGAROException
from figaro.mixture import mixture
from figaro.cosmology import Planck18, Planck15, dVdz_approx_planck18, dVdz_approx_planck15
from pathlib import Path
from tqdm import tqdm

supported_extensions = ['h5', 'hdf5', 'hdf', 'txt', 'dat', 'csv', 'json']
supported_waveforms  = ['combined', 'imr', 'seob']
injected_pars        = ['m1', 'm2', 'z', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z', 'ra', 'dec']
loadable_inj_pars    = injected_pars + ['q', 'chi_eff', 'chi_p', 's1', 's2', 'luminosity_distance', 'log_z', 'm1_detect', 'm2_detect']
mass_parameters      = ['m1', 'm2', 'm1_detect', 'm2_detect', 'mc', 'mt', 'q']
spin_parameters      = ['s1x', 's1y', 's1z', 's2x', 's2y', 's2z', 's1', 's2', 'chi_eff', 'chi_p']
detector_parameters  = ['m1_detect', 'm2_detect']

GW_par = {'m1'                 : 'mass_1_source',
          'm2'                 : 'mass_2_source',
          'm1_detect'          : 'mass_1',
          'm2_detect'          : 'mass_2',
          'mc'                 : 'chirp_mass_source',
          'mc_detect'          : 'chirp_mass',
          'mt'                 : 'total_mass_source',
          'z'                  : 'redshift',
          'log_z'              : 'log_redshift',
          'q'                  : 'mass_ratio',
          'eta'                : 'symmetric_mass_ratio',
          'chi_eff'            : 'chi_eff',
          'chi_p'              : 'chi_p',
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
          'phi_12'             : 'phi_12',
          'psi'                : 'psi',
          'cos_iota'           : 'cos_iota',
          'phase'              : 'phase',
          'tc'                 : 'geocent_time',
          'snr'                : 'network_matched_filter_snr',
          'far'                : 'far',
          'log_prior'          : 'log_prior',
          'log_likelihood'     : 'log_likelihood',
          }

# LVK 03 injections have a slightly different nomenclature (no underscore)
# See https://zenodo.org/records/7890437
inj_par = copy.deepcopy(GW_par)
inj_par['m1']                  = 'mass1_source'
inj_par['m2']                  = 'mass2_source'
inj_par['m1_detect']           = 'mass1'
inj_par['m2_detect']           = 'mass2'
inj_par['s1x']                 = 'spin1x'
inj_par['s1y']                 = 'spin1y'
inj_par['s1z']                 = 'spin1z'
inj_par['s2x']                 = 'spin2x'
inj_par['s2y']                 = 'spin2y'
inj_par['s2z']                 = 'spin2z'
inj_par['luminosity_distance'] = 'distance'

supported_pars = [p for p in GW_par.keys() if p not in ['snr', 'far']]

def available_gw_pars():
    """
    Print a list of available GW parameters
    """
    print(supported_pars)

class openfile:
    def __init__(self, file):
        self.file = Path(file)
    
    def __enter__(self):
        if self.file.suffix == '.json':
            self.openfile = open(self.file)
            return json.load(self.openfile)
        else:
            self.openfile = h5py.File(self.file, 'r')
            return self.openfile
    
    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.openfile.close()

def load_single_event(event, seed = False, par = None, n_samples = -1, cosmology = 'Planck18', volume = False, waveform = 'combined', snr_threshold = None, far_threshold = None, likelihood = False):
    '''
    Loads the data from .txt/.dat files (for simulations) or .h5/.hdf5 files (posteriors from GWTC) for a single event.
    Default cosmological parameters from Planck Collaboration (2021) in a flat Universe (https://www.aanda.org/articles/aa/pdf/2020/09/aa33910-18.pdf)
    Not all GW parameters are implemented: run figaro.load.available_gw_pars() for a list of the available parameters.
    
    Arguments:
        str or Path file:     file with samples
        bool seed:            fixes the seed to a default value (1) for reproducibility
        list-of-str par:      list with parameter(s) to extract from GW posteriors (m1, m2, mc, z, chi_effective)
        int n_samples:        number of samples for (random) downsampling. Default -1: all samples
        str cosmology:        set of cosmological parameters (Planck18 or Planck15)
        bool volume:          if True, loads RA, dec and Luminosity distance (for skymaps)
        str waveform:         waveform family to be used ('combined', 'seob', 'imr')
        double snr_threhsold: SNR threshold for event filtering. For injection analysis only.
        double far_threshold: FAR threshold for event filtering. For injection analysis only.
        bool likelihood:      resample to get likelihood samples
        
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
    if ext not in supported_extensions:
        raise TypeError("File {0}.{1} is not supported".format(name, ext))
    if ext not in ['h5', 'hdf5', 'json']:
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
            wrong_pars = [p for p in par if p not in GW_par.keys()]
            raise FIGAROException("The following parameters are not implemented: "+", ".join(wrong_pars)+". Run figaro.load.available_gw_pars() for a list of available parameters.")
        # If everything is ok, load the samples
        else:
            out = _unpack_gw_posterior(event, par = par, n_samples = n_samples, cosmology = cosmology, rdstate = rdstate, waveform = waveform, snr_threshold = snr_threshold, far_threshold = far_threshold, likelihood = likelihood)
    
    if out is None:
        return out, name
    
    if len(np.shape(out)) == 1:
        out = np.atleast_2d(out).T
    return out, name

def load_data(path, seed = False, par = None, n_samples = -1, cosmology = 'Planck18', volume = False, waveform = 'combined', snr_threshold = None, far_threshold = None, verbose = True, likelihood = False):
    '''
    Loads the data from .txt files (for simulations) or .h5/.hdf5/.dat files (posteriors from GWTC-x).
    Default cosmological parameters from Planck Collaboration (2021) in a flat Universe (https://www.aanda.org/articles/aa/pdf/2020/09/aa33910-18.pdf)
    Not all GW parameters are implemented: run figaro.load.available_gw_pars() for a list of available parameters.
    
    Arguments:
        str or Path path:     folder with data files
        bool seed:            fixes the seed to a default value (1) for reproducibility
        list-of-str par:      list with parameter(s) to extract from GW posteriors
        int n_samples:        number of samples for (random) downsampling. Default -1: all samples
        str cosmology:        set of cosmological parameters (Planck18 or Planck15)
        str waveform:         waveform family to be used ('combined', 'seob', 'imr')
        double snr_threhsold: SNR threshold for event filtering. For injection analysis only.
        double far_threshold: FAR threshold for event filtering. For injection analysis only.
        bool verbose:         show progress bar
        bool likelihood:      resample to get likelihood samples

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
        if ext not in supported_extensions:
            raise TypeError("File {0}.{1} is not supported".format(name, ext))
        if ext not in ['h5', 'hdf5', 'json']:
            if par is not None:
                warnings.warn("Par names (or volume keyword) are ignored for .txt/.dat/.csv files")
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
                wrong_pars = [p for p in par if p not in GW_par.keys()]
                raise FIGAROException("The following parameters are not implemented: "+", ".join(wrong_pars)+". Run figaro.load.available_gw_pars() for a list of available parameters.")
            # If everything is ok, load the samples
            else:
                out = _unpack_gw_posterior(event, par = par, n_samples = n_samples, cosmology = cosmology, rdstate = rdstate, waveform = waveform, snr_threshold = snr_threshold, far_threshold = far_threshold, likelihood = likelihood)
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

def _unpack_gw_posterior(event, par, cosmology, rdstate, n_samples = -1, waveform = 'combined', snr_threshold = None, far_threshold = None, likelihood = False):
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
        str event:                 file to read
        list-of-str par:           parameter to extract
        str cosmology:             set of cosmological parameters to use.
        np.random.rdstate rdstate: state for random number generation
        int n_samples:             number of samples for (random) downsampling. Default -1: all samples
        str waveform:              waveform family to be used ('combined', 'imr', 'seob')
        double snr_threhsold:      SNR threshold for event filtering. For injection analysis only.
        double far_threshold:      FAR threshold for event filtering. For injection analysis only.
        bool likelihood:           resample to get likelihood samples
    
    Returns:
        np.ndarray: samples
    '''
    if cosmology == 'Planck18':
        omega = Planck18
    elif cosmology == 'Planck15':
        omega = Planck15
    else:
        raise FIGAROException("Cosmology not supported")
    if waveform not in supported_waveforms:
        raise FIGAROException("Unknown waveform: please use 'combined' (default), 'imr' or 'seob'")
    
    if far_threshold is not None and snr_threshold is not None:
        warnings.warn("Both FAR and SNR threshold provided. FAR will be used.")
        snr_threshold = None
    
    if far_threshold is not None:
        if 'far' not in par:
            par = np.append(par, 'far')
    elif snr_threshold is not None:
        if 'snr' not in par:
            par = np.append(par, 'snr')
    
    with openfile(event) as f:
        samples     = []
        loaded_pars = []
        flag_filter = False
        try:
            try:
                # LVK injections
                try:
                    data = f['MDC']['posterior_samples']
                except:
                    try:
                        # Nessai .json file
                        data = f['posterior']['content']
                    except:
                        # Playground
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
                    elif name == 'log_z':
                        samples.append(np.log(data['redshift']))
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
            if likelihood:
                inv_prior = 1./_prior_gw(par, data, cosmology)
                h         = np.random.uniform(0,1, len(samples))
                samples   = samples[h < inv_prior/np.max(inv_prior)]
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
            ss['log_z']   = np.log(ss['z'])
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
            if likelihood:
                inv_prior = 1./_prior_gw(par, ss, cosmology)
                h         = np.random.uniform(0, np.max(inv_prior), len(samples))
                samples   = samples[h < inv_prior]
            if n_samples > -1:
                s = int(min([n_samples, len(samples)]))
                return samples[rdstate.choice(np.arange(len(samples)), size = s, replace = False)]
            else:
                return samples

def _prior_gw(par, samples, cosmology = 'Planck15', uniform_dVdz = True):
    """
    GW prior parameters, following https://docs.ligo.org/RatesAndPopulations/gwpopulation_pipe/_modules/gwpopulation_pipe/data_collection.html#evaluate_prior
    
    Arguments:
        np.ndarray samples: posterior samples
        list-of-str par:    parameter(s)
        str cosmology:      cosmological parameters
        bool uniform_dVdz:  assumes a uniform in dVdz prior for z
    
    Returns:
        np.ndarray: prior values
    """
    if cosmology == 'Planck18':
        omega = Planck18
        dVdz  = dVdz_approx_planck18
    elif cosmology == 'Planck15':
        omega = Planck15
        dVdz  = dVdz_approx_planck15
    else:
        raise FIGAROException("Cosmology not supported")
    vol    = omega.ComovingVolume(2.3)/1e9
    DL_max = omega.LuminosityDistance(2.3)
    prior  = np.ones(len(samples[GW_par['z']]))
    # Redshift prior (uniform in comoving source frame)
    if ('z' in par) or ('log_z' in par) or ('luminosity_distance' in par):
        if uniform_dVdz:
            prior *= dVdz(np.array(samples[GW_par['z']]))/((1.+np.array(samples[GW_par['z']]))*vol)
        else:
            prior *= (np.array(samples[GW_par['luminosity_distance']])**2/DL_max**3)*omega.dDLdz(np.array(samples[GW_par['z']]))
        if 'luminosity_distance' in par:
            prior /= omega.dDLdz(np.array(samples[GW_par['z']]))
        if 'log_z' in par:
            prior *= samples[GW_par['z']]
    # Mass prior (uniform in detector-frame component masses)
    n_mass_pars = np.sum([item in par for item in ['m1','m2','mc','q']])
    if n_mass_pars > 0:
        prior *= (1+np.array(samples[GW_par['z']]))**np.min([n_mass_pars, 2])
    if 'q' in par:
        prior *= np.array(samples[GW_par['m1']])/(1+np.array(samples[GW_par['z']]))
    if ('mc' in par or 'mc_detect' in par):
        prior *= (1 + np.array(samples[GW_par['q']])) ** 0.2 / np.array(samples[GW_par['q']]) ** 0.6
    return prior
    
def save_density(draws, folder = '.', name = 'density', ext = 'json'):
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
                if 'components' in density.keys():
                    density.pop('components')
                for key in density.keys():
                    value = density[key]
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    else:
                        value = float(value)
                    density[key] = value
            ll.append(list_of_dicts)
        s = json.dumps(ll)
        with open(Path(folder, name + '.json'), 'w') as f:
            json.dump(s, f)
    else:
        raise FIGAROException("Extension {0} is not supported. Valid extensions are pkl or json.")

def load_density(path, make_comp = True):
    """
    Loads a list of figaro.mixture instances from path.
    If the requested file extension (pkl or json) is not available, it tries loading the other.

    Arguments:
        str or Path path: path with draws (file or folder)
        bool make_comp:   make component objects

    Returns:
        list: figaro.mixture object instances
    """
    path = Path(path)
    if path.is_file():
        return _load_density_file(path, make_comp)
    else:
        files = [_load_density_file(file, make_comp) for file in path.glob('*.[jp][sk][ol]*') if not file.stem == 'posteriors_single_event']
        if len(files) > 0:
            return files
        else:
            raise FIGAROException("Density file(s) not found")

def _load_density_file(file, make_comp = True):
    """
    Loads a list of figaro.mixture instances from file.
    If the requested file extension (pkl or json) is not available, it tries loading the other.

    Arguments:
        str or Path file: file with draws
        bool make_comp:   make component objects

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
                return _load_json(file.with_suffix('.json'), make_comp)
            except FileNotFoundError:
                raise FIGAROException("{0} not found. Please provide it or re-run the inference.".format(file.name))
    elif ext == '.json':
        try:
            return _load_json(file, make_comp)
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

def _load_json(file, make_comp = True):
    """
    Loads a list of figaro.mixture instances from json file

    Arguments:
        str or Path file: file with draws
        bool make_comp:   make component objects

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
                if key == 'probit':
                    dict_[key] = bool(value)
            draws.append(mixture(**dict_, make_comp = make_comp))
        ll.append(draws)
    if len(ll) == 1:
        return ll[0]
    return ll

def load_selection_function(file, par = None, far_threshold = 1, snr_threshold = 10, cosmology = 'Planck15'):
    """
    Loads the selection function, either from a python module containing a method called 'selection_function' or via injections.
    If injections, it assumes that the last column of a txt/csv/dat file contains the sampling pdf.
    
    Arguments:
        str or Path file: selection function file
        list-of-str par:  list with parameter(s) to extract from GW posteriors
        double far_threshold: FAR threshold to filter LVK injections
        double snr_threshold: SNR threshold to filter LVK injections

    
    Returns:
        np.ndarray or callable: detected samples or callable with approximant
        np.ndarray or NoneType: injection pdf (for samples) or None (for approximant)
        int or NoneType:        total number of injections
        double duration:        duration of the observation
    """
    file = Path(file)
    ext  = file.parts[-1].split('.')[1]
    if ext not in supported_extensions + ['py']:
        raise FIGAROException("Selection function file not supported")
    if ext == 'py':
        selfunc_file_name = file.parts[-1].split('.')[0]
        spec              = importlib.util.spec_from_file_location(selfunc_file_name, file)
        selfunc_module    = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(selfunc_module)
        selfunc           = selfunc_module.selection_function
        inj_pdf           = None
        n_total_inj       = None
        try:
            duration      = selfunc_module.duration
        except:
            duration      = 1.
    else:
        if ext not in ['h5','hdf5']:
            samples     = np.loadtxt(file)
            det_idx     = samples[:,-1]
            selfunc     = samples[:,:-2][det_idx == 1]
            inj_pdf     = samples[:,-2][det_idx == 1]
            n_total_inj = len(samples)
            duration    = 1.
        else:
            selfunc, inj_pdf, n_total_inj, duration = _unpack_injections(file, par, far_threshold, snr_threshold, cosmology)
    return selfunc, inj_pdf, n_total_inj, duration

def _unpack_injections(file, par, far_threshold = 1., snr_threshold = 10, cosmology = 'Planck15'):
    """
    Reads data from .h5/.hdf5 injection file (https://zenodo.org/records/7890437).
    A sample is considered detected if at least one of the searches calls a detection.
    
    Arguments:
        str event:            file to read
        str par:              parameter to extract
        double far_threshold: FAR threshold for injection filtering
        double snr_threshold: SNR threshold for injection filtering
    
    Returns:
        np.ndarray:      samples
        np.ndarray:      injection pdf
        int or NoneType: total number of injections
        double duration: duration of the observation
    """
    if cosmology == 'Planck18':
        omega = Planck18
    elif cosmology == 'Planck15':
        omega = Planck15
    # Check that a list of parameters is passed
    if par is None:
        raise TypeError("Please provide a list of parameter names you want to load (e.g. ['m1']).")
    # Check that all the parametes are loadable
    if not np.all([p in loadable_inj_pars for p in par]):
        wrong_pars = [p for p in par if p not in loadable_inj_pars]
        raise FIGAROException("The following parameters are not implemented: "+", ".join(wrong_pars))
    with h5py.File(file, 'r') as f:
        data          = f['injections']
        joint_dataset = 'name' in data.keys()
        n_total_inj   = int(data.attrs['total_generated'])
        duration      = data.attrs['analysis_time_s']/(60.*60.*24.*365) # Years
        try:
            far_idx = np.zeros(np.array(data['far_cwb']).shape, dtype = bool)
        except KeyError:
            far_idx = np.zeros(np.array(data['snr']).shape, dtype = bool)
        for key in data.keys():
            if 'ifar' in key.lower():
                far_idx |= data[key][()] > 1./far_threshold
        if joint_dataset:
            # O1+O2+O3
            names = np.array(data['name'], dtype = str)
            snr   = np.array(data['optimal_snr_net'])
            idx   = np.where(names == 'o3', far_idx, snr > snr_threshold)
        else:
            if np.sum(far_idx) > 0:
                # O3 only
                idx = np.where(far_idx, True, False)
            else:
                # Simulated dataset (SNR filter)
                snr = np.array(data['snr'])
                idx = np.where(snr > snr_threshold, True, False)
        # Load samples
        samples = np.zeros((len(par), np.sum(idx)))
        inj_pdf = np.ones(np.sum(idx))
        # Parameters
        m1  = np.array(data[inj_par['m1']])[idx]
        m2  = np.array(data[inj_par['m2']])[idx]
        q   = m2/m1
        z   = np.array(data[inj_par['z']])[idx]
        try:
            s1x = np.array(data[inj_par['s1x']])[idx]
            s1y = np.array(data[inj_par['s1y']])[idx]
            s1z = np.array(data[inj_par['s1z']])[idx]
            s2x = np.array(data[inj_par['s2x']])[idx]
            s2y = np.array(data[inj_par['s2y']])[idx]
            s2z = np.array(data[inj_par['s2z']])[idx]
        except:
            pass
        for i, lab in enumerate(par):
            # Already available parameters:
            if lab in injected_pars:
                if not lab == 'cos_theta_jn':
                    samples[i] = np.array(data[inj_par[lab]])[idx]
                else:
                    samples[i] = np.cos(np.array(data[inj_par[lab]])[idx])
            else:
                # Masses
                if lab == 'q':
                    samples[i] = q
                if lab == 'mt':
                    samples[i] = m1 + m2
                if lab == 'mc':
                    samples[i] = m1*m2**(3./5.)/(m1+m2)**(1./5.)
                if lab == 'm1_detect':
                    samples[i] = m1*(1+z)
                if lab == 'm2_detect':
                    samples[i] = m2*(1+z)
                # Spins
                if lab == 's1':
                    samples[i] = np.sqrt(s1x**2+s1y**2+s1z**2)
                if lab == 's2':
                    samples[i] = np.sqrt(s2x**2+s2y**2+s2z**2)
                if lab == 'chi_eff':
                    samples[i] = (s1z + q*s2z)/(1+q)
                if lab == 'chi_p':
                    samples[i] = np.maximum(np.sqrt(s1x**2+s1y**2), np.sqrt(s2x**2+s2y**2)*q*(4*q+3)/(4+3*q))
                # Distance
                if lab == 'luminosity_distance':
                    samples[i] = omega.LuminosityDistance(z)
                if lab == 'log_z':
                    samples[i] = np.log(z)
        # Sampling pdf
        n_mass_pars = len([lab for lab in par if lab in mass_parameters])
        n_det_pars  = len([lab for lab in par if lab in detector_parameters])
        n_spin_pars = len([lab for lab in par if lab in spin_parameters])
        if joint_dataset:
            if not (('z' in par) and (n_mass_pars == 2)):
                raise FIGAROException("Cannot unpack individual parameter sampling PDF for combined injections")
            inj_pdf  = np.array(data['sampling_pdf'])[idx]
            # Remove spins if not needed
            spin_pdf = 1./(4*np.pi*(s1x**2+s1y**2+s1z**2))*1./(4*np.pi*(s2x**2+s2y**2+s2z**2))
            inj_pdf /= spin_pdf**((6-n_spin_pars)/6.)
        else:
            inj_pdf = np.ones(np.sum(idx))
            # Masses
            if n_mass_pars == 1:
                inj_pdf *= np.array(data['mass1_source_sampling_pdf'])[idx]
            elif n_mass_pars == 2:
                inj_pdf *= np.array(data['mass1_source_mass2_source_sampling_pdf'])[idx]
            # Spins
            if n_spin_pars > 0:
                inj_pdf *= (np.array(data['spin1x_spin1y_spin1z_sampling_pdf'])[idx]*np.array(data['spin2x_spin2y_spin2z_sampling_pdf'])[idx])**(n_spin_pars/6.)
            # Distance
            if 'z' in par:
                inj_pdf *= np.array(data['redshift_sampling_pdf'])[idx]
            if 'log_z' in par:
                inj_pdf *= np.array(data['redshift_sampling_pdf'])[idx]*np.array(data[inj_par['z']])[idx]
            if 'luminosity_distance' in par:
                try:
                    inj_pdf *= np.array(data['luminosity_distance_sampling_pdf'])[idx]
                except:
                    inj_pdf *= np.array(data['redshift_sampling_pdf'])[idx]/omega.dDLdz(np.array(data[inj_par['z']])[idx])
            # Sky position
            if 'ra' in par:
                inj_pdf *= np.array(data['right_ascension_sampling_pdf'])[idx]
            if 'dec' in par:
                inj_pdf *= np.array(data['declination_sampling_pdf'])[idx]
        # Change of variable
        if 'q' in par:
            inj_pdf *= m1
        if 's1' in par or 'chi_eff' in par or 'chi_p' in par:
            inj_pdf *= 2*np.pi*(s1x**2+s1y**2+s1z**2)
        if 's2' in par or 'chi_eff' in par or 'chi_p' in par:
            inj_pdf *= 2*np.pi*(s2x**2+s2y**2+s2z**2)
        if n_det_pars > 0:
            inj_pdf /= (1+z)**n_det_pars

#    import matplotlib.pyplot as plt
#    plt.hist(np.log(inj_pdf))
#    plt.show()
    return samples.T, inj_pdf, n_total_inj, duration
