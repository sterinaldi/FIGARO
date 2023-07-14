import numpy as np
import h5py
import re
import warnings

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from corner import corner
import imageio

from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy.io import fits
from astropy.wcs import WCS
import pyvo as vo
import socket

from scipy.special import logsumexp
from scipy.stats import multivariate_normal as mn
from numba import njit, prange
import dill

from figaro.mixture import DPGMM
from figaro.transform import *
from figaro.coordinates import celestial_to_cartesian, cartesian_to_celestial, inv_Jacobian
from figaro.credible_regions import ConfidenceArea, ConfidenceVolume, FindNearest_Volume, FindLevelForHeight
from figaro.diagnostic import compute_entropy_single_draw, angular_coefficient
from figaro.exceptions import FIGAROException
from figaro import plot_settings

try:
    from figaro.cosmology import CosmologicalParameters
    lal_flag = True
except ModuleNotFoundError:
    warnings.warn("LAL is not installed. If provided, galaxy catalog will not be loaded")
    lal_flag = False

from pathlib import Path
from distutils.spawn import find_executable
from tqdm import tqdm

@njit
def log_add(x, y):
    """
    Compute log(np.exp(x) + np.exp(y))
    
    Arguments:
        double x: first addend (log)
        double y: second addend (log)
    
    Returns:
        double: log(np.exp(x) + np.exp(y))
    """
    if x >= y:
        return x+np.log1p(np.exp(y-x))
    else:
        return y+np.log1p(np.exp(x-y))

@njit
def log_add_array(x,y):
    """
    Compute log(np.exp(x) + np.exp(y)) element-wise
    
    Arguments:
        np.ndarray x: first addend (log)
        np.ndarray y: second addend (log)
    
    Returns:
        np.ndarray: log(np.exp(x) + np.exp(y)) element-wise
    """
    res = np.zeros(len(x), dtype = np.float64)
    for i in prange(len(x)):
        res[i] = log_add(x[i],y[i])
    return res

# natural sorting.
# list.sort(key = natural_keys)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    list.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class VolumeReconstruction(DPGMM):
    """
    Class to reconstruct the 3D probability density given a set of samples or during a PE run.
    Child class of DPGMM class (defined in mixture.py)
    
    Arguments:
        double max_dist:          maximum distance (in Mpc)
        str or Path out_folder:   output folder
        tuple prior_pars:         NIW prior parameters (k, L, nu, mu - see https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf)
        double alpha0:            initial guess for concentration parameter
        list-of-int n_gridpoints: number of gridpoints for each dimension (ra, dec, dist). Default is set to have a 0.25 deg^2 resolution
        str name:                 name to be given to output folders and files
        list-of-str labels:       plot labels
        list-of-double levels:    credible levels (between 0 and 1)
        bool latex:               if True, LaTeX is used for plots (if available)
        bool incr_plot:           if True, produce plots at fixed number of samples. Otherwise, skymaps are produced at the end of the inference only
        str or Path glade_file:   file containing GLADE+ (hdf5 file created using the create_glade.py pipeline)
        dict cosmology:           dictionary containing h = H0/100 km/(s*Mpc), om (matter density parameter) and ol (cosmological constant density). Default from Planck (2021)
        int n_gal_to_plot:        number of galaxies to include in galaxy plots
        double region_to_plot:    credible region to plot
        bool entropy:             use entropy to assess convergence and begin producing skymaps
        list-of-double true_host: position of the true host, if known
        str host_name:            host name, for plotting purposes
        int entropy_step:         interval (expressed as number of draws) between two subsequent entropy evaluations
        int entropy_ac_step:      window lenght for entropy angular coefficient
        int n_sign_changes:       number of zero-crossings before beginning producing skymaps (requires entropy = True)
        bool virtual_observatory: places the virtual observatory fov behind galaxy plots. Warning: under development, might not work
    
    Returns:
        VolumeReconstruction: instance of VolumeReconstruction class
    """
    def __init__(self, max_dist,
                       out_folder          = '.',
                       prior_pars          = (1e-3, np.identity(3)*0.01**2, 10, np.zeros(3)),
                       alpha0              = 1,
                       n_gridpoints        = [720, 360, 100], # RA, dec, DL
                       name                = 'skymap',
                       labels              = ['$\\alpha$', '$\\delta$', '$D\ [Mpc]$'],
                       levels              = [0.50, 0.90],
                       latex               = False,
                       incr_plot           = False,
                       glade_file          = None,
                       cosmology           = {'h': 0.674, 'om': 0.315, 'ol': 0.685},
                       n_gal_to_plot       = -1,
                       region_to_plot      = 0.9,
                       entropy             = False,
                       n_entropy_MC_draws  = 1e3,
                       true_host           = None,
                       host_name           = 'Host',
                       entropy_step        = 1,
                       entropy_ac_step     = 500,
                       n_sign_changes      = 5,
                       virtual_observatory = False,
                       ):
                
        self.max_dist = max_dist
        bounds = np.array([[-max_dist, max_dist] for _ in range(3)])
        self.volume_already_evaluated = False
        
        super().__init__(bounds, prior_pars, alpha0)
        
        if incr_plot:
            self.next_plot = 20
        else:
            self.next_plot = np.inf
            
        if latex:
            if find_executable('latex'):
                rcParams["text.usetex"] = True
        self.latex = latex
        
        # Grid
        self.ra   = np.linspace(0,2*np.pi, n_gridpoints[0])
        self.dec  = np.linspace(-np.pi/2, np.pi/2., n_gridpoints[1])
        self.dist = np.linspace(0, max_dist, n_gridpoints[2])
        self.dD   = np.diff(self.dist)[0]
        self.dra  = np.diff(self.ra)[0]
        self.ddec = np.diff(self.dec)[0]
        # For loops
        grid = []
        measure_3d = []
        distance_measure_3d = []
        for ra_i in self.ra:
            for dec_i in self.dec:
                cosdec = np.cos(dec_i)
                for d_i in self.dist:
                    grid.append(np.array([ra_i, dec_i, d_i]))
                    measure_3d.append(cosdec*d_i**2)
                    distance_measure_3d.append(d_i**2)
        self.grid = np.array(grid)
        self.log_measure_3d = np.log(measure_3d).reshape(len(self.ra), len(self.dec), len(self.dist))
        self.distance_measure_3d = np.array(distance_measure_3d).reshape(len(self.ra), len(self.dec), len(self.dist))
        grid2d = []
        measure_2d = []
        for ra_i in self.ra:
            for dec_i in self.dec:
                grid2d.append(np.array([ra_i, dec_i]))
                measure_2d.append(np.cos(dec_i))
        self.grid2d = np.array(grid2d)
        self.log_measure_2d = np.log(measure_2d).reshape(len(self.ra), len(self.dec))
        # Meshgrid
        self.ra_2d, self.dec_2d = np.meshgrid(self.ra, self.dec)
        self.cartesian_grid = celestial_to_cartesian(self.grid)
        self.probit_grid = transform_to_probit(self.cartesian_grid, self.bounds)

        # Jacobians
        self.log_inv_J = -np.log(inv_Jacobian(self.grid)) - probit_logJ(self.probit_grid, self.bounds)
        self.inv_J = np.exp(self.log_inv_J)
        
        # True host
        if true_host is not None:
            if len(true_host) == 2:
                self.true_host = np.concatenate((np.array(true_host), np.ones(1)))
            elif len(true_host) == 3:
                self.true_host = true_host
        else:
            self.true_host = true_host
        self.host_name = host_name
        if self.true_host is not None:
            self.pixel_idx  = FindNearest_Volume(self.ra, self.dec, self.dist, self.true_host)
            self.true_pixel = np.array([self.ra[self.pixel_idx[0]], self.dec[self.pixel_idx[1]], self.dist[self.pixel_idx[2]]])
        
        # Credible regions levels
        self.levels      = np.array(levels)
        self.areas_N     = {cr:[] for cr in self.levels}
        self.volumes_N   = {cr:[] for cr in self.levels}
        self.N           = []
        self.flag_skymap = True
        if entropy == True:
            self.flag_skymap = False
        
        # Catalog
        self.catalog = None
        if lal_flag and glade_file is not None:
            self.cosmology = CosmologicalParameters(cosmology['h'], cosmology['om'], cosmology['ol'], 1, 0)
            self.load_glade(glade_file)
            self.cartesian_catalog = celestial_to_cartesian(self.catalog)
            self.probit_catalog    = transform_to_probit(self.cartesian_catalog, self.bounds)
            self.log_inv_J_cat     = -np.log(inv_Jacobian(self.catalog)) - probit_logJ(self.probit_catalog, self.bounds)
            self.inv_J_cat         = np.exp(self.log_inv_J)
        if n_gal_to_plot == -1 and self.catalog is not None:
            self.n_gal_to_plot = len(self.catalog)
        else:
            self.n_gal_to_plot = n_gal_to_plot
        if region_to_plot in self.levels:
            self.region = region_to_plot
        else:
            self.region = self.levels[0]
        self.virtual_observatory = virtual_observatory
        
        # Entropy
        self.entropy            = entropy
        self.entropy_step       = entropy_step
        self.entropy_ac_step    = entropy_ac_step
        self.N_for_ac           = np.arange(self.entropy_ac_step)*self.entropy_step
        self.n_entropy_MC_draws = int(n_entropy_MC_draws)
        self.R_S                = []
        self.ac                 = []
        self.n_sign_changes     = n_sign_changes
        
        # Output
        self.name       = name
        self.labels     = labels
        self.out_folder = Path(out_folder).resolve()
        if not self.out_folder.exists():
            self.out_folder.mkdir()
        self.make_folders()
    
    def initialise(self, true_host = None, out_folder = None, incr_plot = None, entropy = None, name = None):
        """
        Initialise the mixture to initial conditions to analyse a new event.
        
        Arguments:
            list-of-doubles true_host: true host for a new GW event
            str or Path out_folder:    new output folder
        """
        self.volume_already_evaluated = False
        super().initialise()
        self.true_host   = true_host
        self.R_S         = []
        self.ac          = []
        self.areas_N     = {cr:[] for cr in self.levels}
        self.volumes_N   = {cr:[] for cr in self.levels}
        self.N           = []
        self.flag_skymap = True
        if name is not None:
            self.name = name
        if entropy is not None:
            self.entropy = entropy
        if self.entropy == True:
            self.flag_skymap = False
        if self.true_host is not None:
            self.pixel_idx  = FindNearest_Volume(self.ra, self.dec, self.dist, self.true_host)
            self.true_pixel = np.array([self.ra[self.pixel_idx[0]], self.dec[self.pixel_idx[1]], self.dist[self.pixel_idx[2]]])
        if incr_plot is not None:
            if incr_plot:
                self.next_plot = 20
            else:
                self.next_plot = np.inf
        else:
            if self.next_plot < np.inf:
                self.next_plot = 20
        if out_folder is not None:
            self.out_folder = Path(out_folder).resolve()
            if not self.out_folder.exists():
                self.out_folder.mkdir()
        self.make_folders()
        
    def load_glade(self, glade_file):
        """
        Load GLADE+ from hdf5 file.
        This is tailored to the GLADE+ hdf5 file created by the create_glade.py pipeline.
        
        Arguments:
            str or Path glade_file: glade file to be uploaded
        """
        self.glade_header =  ' '.join(['ra', 'dec', 'z', 'm_B', 'm_K', 'm_W1', 'm_bJ', 'logp'])
        with h5py.File(glade_file, 'r') as f:
            ra  = np.array(f['ra'])
            dec = np.array(f['dec'])
            z   = np.array(f['z'])
            B   = np.array(f['m_B'])
            K   = np.array(f['m_K'])
            W1  = np.array(f['m_W1'])
            bJ  = np.array(f['m_bJ'])
        DL = self.cosmology.LuminosityDistance(z)
        catalog = np.array([ra, dec, DL]).T
        self.catalog = catalog[catalog[:,2] < self.max_dist]
        catalog_with_mag = np.array([ra, dec, z, B, K, W1, bJ]).T
        self.catalog_with_mag = catalog_with_mag[catalog[:,2] < self.max_dist]

    def make_folders(self):
        """
        Make folders for outputs
        """
        self.skymap_folder = Path(self.out_folder, 'skymaps', self.name)
        if not self.skymap_folder.exists():
            self.skymap_folder.mkdir(parents=True)
        if self.catalog is not None:
            self.volume_folder = Path(self.out_folder, 'volume', self.name)
            if not self.volume_folder.exists():
                self.volume_folder.mkdir(parents=True)
            self.catalog_folder = Path(self.out_folder, 'catalogs', self.name)
            if not self.catalog_folder.exists():
                self.catalog_folder.mkdir(parents=True)
        if self.next_plot < np.inf:
            self.CR_folder = Path(self.out_folder, 'CR')
            if not self.CR_folder.exists():
                self.CR_folder.mkdir()
            self.gif_folder = Path(self.out_folder, 'gif')
            if not self.gif_folder.exists():
                self.gif_folder.mkdir()
        if self.entropy:
            self.entropy_folder = Path(self.out_folder, 'entropy')
            if not self.entropy_folder.exists():
                self.entropy_folder.mkdir()
        self.density_folder = Path(self.out_folder, 'density')
        if not self.density_folder.exists():
            self.density_folder.mkdir()


    def _pdf_probit(self, x):
        """
        Evaluate mixture at point(s) x in probit space.
        Overwrites parent method to avoid memory issues in 3D grid or catalog evaluation
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at (in probit space)
        
        Returns:
            np.ndarray: mixture.pdf(x)
        """
        p = np.zeros(len(x))
        for comp, wi in zip(self.mixture, self.w):
            p += wi*mn(comp.mu, comp.sigma).pdf(x)
        return p
    
    def _logpdf_probit(self, x):
        """
        Evaluate log mixture at point(s) x in probit space.
        Overwrites parent method to avoid memory issues in 3D grid or catalog evaluation
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at (in probit space)
        
        Returns:
            np.ndarray: mixture.logpdf(x)
        """
        p = -np.ones(len(x))*np.inf
        for comp, wi in zip(self.mixture, self.log_w):
            p = log_add_array(p, wi + mn(comp.mu, comp.sigma).logpdf(x))
        return p

    def add_new_point(self, x):
        """
        Update the probability density reconstruction adding a new sample
        Sample must be in celestial coordinate and in the following order: [ra, dec, dist].
        
        Arguments:
            np.ndarray x: sample
        """
        self.volume_already_evaluated = False
        cart_x = celestial_to_cartesian(x)
        super().add_new_point(cart_x)
        if self.flag_skymap and self.n_pts == self.next_plot:
            self.N.append(self.next_plot)
            self.next_plot = 2*self.next_plot
            self.make_skymap()
            self.make_volume_map(n_gals = self.n_gal_to_plot)
    
    def sample_from_volume(self, n_samps):
        """
        Draw samples from volume
        
        Arguments:
            int n_samps: number of samples to draw
        
        Returns:
            np.ndarray: samples in celestial coordinates (ra, dec, D)
        """
        samples = self.rvs(n_samps)
        return cartesian_to_celestial(samples)
    
    def plot_samples(self, n_samps, initial_samples = None):
        """
        Plot samples from DPGMM reconstruction along with PE samples (if available)
        
        Arguments:
            int n_samps:                number of samples to draw
            np.ndarray initial_samples: PE samples
        """
        mix_samples = self.sample_from_volume(n_samps)
        if initial_samples is not None:
            if self.true_host is not None:
                c = corner(initial_samples, color = 'coral', labels = self.labels, truths = self.true_host, hist_kwargs={'density':True, 'label':'$\mathrm{Samples}$'})
            else:
                c = corner(initial_samples, color = 'coral', labels = self.labels, hist_kwargs={'density':True, 'label':'$\mathrm{Samples}$'})
            c = corner(mix_samples, fig = c, color = 'dodgerblue', labels = self.labels, hist_kwargs={'density':True, 'label':'$\mathrm{DPGMM}$'})
        else:
            c = corner(mix_samples, color = 'dodgerblue', labels = self.labels, hist_kwargs={'density':True, 'label':'$\mathrm{DPGMM}$'})
        plt.legend(loc = 0, frameon = False,fontsize = 15, bbox_to_anchor = (1-0.05, 2.8))
        plt.savefig(Path(self.skymap_folder, 'corner_'+self.name+'.pdf'), bbox_inches = 'tight')
        plt.close()
    
    def evaluate_skymap(self):
        """
        Marginalise volume map over luminosity distance to get the 2D skymap and compute credible areas
        """
        if not self.volume_already_evaluated:
            p_vol               = self._pdf_probit(self.probit_grid) * self.inv_J
            self.norm_p_vol     = (p_vol*np.exp(self.log_measure_3d.reshape(p_vol.shape))*self.dD*self.dra*self.ddec).sum()
            self.log_norm_p_vol = np.log(self.norm_p_vol)
            self.p_vol          = p_vol/self.norm_p_vol
            
            # By default computes log(p_vol). If -infs are present, computes log_p_vol
            with np.errstate(divide='raise'):
                try:
                    self.log_p_vol = np.log(self.p_vol)
                except FloatingPointError:
                    self.log_p_vol = self._logpdf_probit(self.probit_grid) + self.log_inv_J - self.log_norm_p_vol
                    
            self.p_vol     = self.p_vol.reshape(len(self.ra), len(self.dec), len(self.dist))
            self.log_p_vol = self.log_p_vol.reshape(len(self.ra), len(self.dec), len(self.dist))
            self.volume_already_evaluated = True

        self.p_skymap = (self.p_vol*self.dD*self.distance_measure_3d).sum(axis = -1)
        
        # By default computes log(p_skymap). If -infs are present, computes log_p_skymap
        with np.errstate(divide='raise'):
            try:
                self.log_p_skymap = np.log(self.p_skymap)
            except FloatingPointError:
                self.log_p_skymap = logsumexp(self.log_p_vol + np.log(self.dD) + np.log(self.distance_measure_3d), axis = -1)

        self.areas, self.skymap_idx_CR, self.skymap_heights = ConfidenceArea(self.log_p_skymap, self.ra, self.dec, log_measure = self.log_measure_2d, adLevels = self.levels)
        for cr, area in zip(self.levels, self.areas):
            self.areas_N[cr].append(area)
    
    def evaluate_volume_map(self):
        """
        Evaluate volume map and compute credbile volumes
        """
        if not self.volume_already_evaluated:
            p_vol               = self._pdf_probit(self.probit_grid) * self.inv_J
            self.norm_p_vol     = (p_vol*np.exp(self.log_measure_3d.reshape(p_vol.shape))*self.dD*self.dra*self.ddec).sum()
            self.log_norm_p_vol = np.log(self.norm_p_vol)
            self.p_vol          = p_vol/self.norm_p_vol
            
            # By default, just uses log(p_vol). If -infs are present, computes log_p_vol
            with np.errstate(divide='raise'):
                try:
                    self.log_p_vol = np.log(self.p_vol)
                except FloatingPointError:
                    self.log_p_vol = self._logpdf_probit(self.probit_grid) + self.log_inv_J - self.log_norm_p_vol

            self.p_vol     = self.p_vol.reshape(len(self.ra), len(self.dec), len(self.dist))
            self.log_p_vol = self.log_p_vol.reshape(len(self.ra), len(self.dec), len(self.dist))
            self.volume_already_evaluated = True
            
        self.volumes, self.idx_CR, self.volume_heights = ConfidenceVolume(self.log_p_vol, self.ra, self.dec, self.dist, log_measure = self.log_measure_3d, adLevels = self.levels)
        
        for cr, vol in zip(self.levels, self.volumes):
            self.volumes_N[cr].append(vol)
    
    def compute_credible_levels_host(self):
        """
        Compute credible levels for true host (if provided)
        """
        self.log_p_vol_host    = self.log_p_vol[self.pixel_idx[0],self.pixel_idx[1],self.pixel_idx[2]]
        self.log_p_skymap_host = self.log_p_skymap[self.pixel_idx[0], self.pixel_idx[1]]
        
        self.CR_host           = FindLevelForHeight(self.log_p_skymap, self.log_p_skymap_host, np.log(self.dra)+np.log(self.ddec))
        self.CV_host           = FindLevelForHeight(self.log_p_vol, self.log_p_vol_host, np.log(self.dra)+np.log(self.ddec)+np.log(self.dD))
        
    
    def evaluate_catalog(self, final_map = False):
        """
        Evaluate the probability of being the host for each entry in the galaxy catalog and rank it accordingly.
        If the inference is finished, save credible areas/volumes.
        
        Arguments:
            bool final_map: flag to raise if the inference is finished
        """
        log_p_cat                  = self._logpdf_probit(self.probit_catalog) + self.log_inv_J_cat - self.log_norm_p_vol
        self.log_p_cat_to_plot     = log_p_cat[np.where(log_p_cat > self.volume_heights[np.where(self.levels == self.region)])]
        self.p_cat_to_plot         = np.exp(self.log_p_cat_to_plot)
        self.cat_to_plot_celestial = self.catalog[np.where(log_p_cat > self.volume_heights[np.where(self.levels == self.region)])]
        self.cat_to_plot_cartesian = self.cartesian_catalog[np.where(log_p_cat > self.volume_heights[np.where(self.levels == self.region)])]
        
        self.sorted_cat = np.c_[self.cat_to_plot_celestial[np.argsort(self.log_p_cat_to_plot)], np.sort(self.log_p_cat_to_plot)][::-1]
        self.sorted_cat_to_txt = np.c_[self.catalog_with_mag[np.where(log_p_cat > self.volume_heights[np.where(self.levels == self.region)])][np.argsort(self.log_p_cat_to_plot)], np.sort(self.log_p_cat_to_plot)][::-1]
        self.sorted_p_cat_to_plot = np.sort(self.p_cat_to_plot)[::-1]
        np.savetxt(Path(self.catalog_folder, self.name+'_{0}'.format(self.n_pts)+'.txt'), self.sorted_cat_to_txt, header = self.glade_header)
        if final_map:
            np.savetxt(Path(self.catalog_folder, 'CR_'+self.name+'.txt'), np.array([self.areas[np.where(self.levels == self.region)], self.volumes[np.where(self.levels == self.region)]]).T, header = 'area volume')
    
    def make_skymap(self, final_map = False):
        """
        Produce skymap.
        
        Arguments:
            bool final_map: flag to raise if the inference is finished
        """
        self.evaluate_skymap()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        c = ax.contourf(self.ra_2d, self.dec_2d, self.p_skymap.T, 500, cmap = 'Reds')
        ax.set_rasterization_zorder(-10)
        c1 = ax.contour(self.ra_2d, self.dec_2d, self.log_p_skymap.T, np.sort(self.skymap_heights), colors = 'black', linewidths = 0.5, linestyles = 'dashed')
        if self.latex:
            ax.clabel(c1, fmt = {l:'{0:.0f}\\%'.format(100*s) for l,s in zip(c1.levels, self.levels[::-1])}, fontsize = 5)
        else:
            ax.clabel(c1, fmt = {l:'{0:.0f}%'.format(100*s) for l,s in zip(c1.levels, self.levels[::-1])}, fontsize = 5)
        for i in range(len(self.areas)):
            c1.collections[i].set_label('${0:.0f}\\%'.format(100*self.levels[-i])+ '\ \mathrm{CR}:'+'{0:.1f}'.format(self.areas[-i]) + '\ \mathrm{deg}^2$')
        handles, labels = ax.get_legend_handles_labels()
        patch = mpatches.Patch(color='grey', label='${0}'.format(self.n_pts)+'\ \mathrm{samples}$', alpha = 0)
        handles.append(patch)
        ax.set_xlabel('$\\alpha$')
        ax.set_ylabel('$\\delta$')
        ax.legend(handles = handles, fontsize = 10, handlelength=0, handletextpad=0, markerscale=0)
        if final_map:
            fig.savefig(Path(self.skymap_folder, self.name+'_all.pdf'), bbox_inches = 'tight')
            if self.next_plot < np.inf:
                fig.savefig(Path(self.gif_folder, self.name+'_all.png'), bbox_inches = 'tight')
        else:
            fig.savefig(Path(self.skymap_folder, self.name+'_{0}'.format(self.n_pts)+'.pdf'), bbox_inches = 'tight')
            if self.next_plot < np.inf:
                fig.savefig(Path(self.gif_folder, self.name+'_{0}'.format(self.n_pts)+'.png'), bbox_inches = 'tight')
        plt.close()
    
    def make_volume_map(self, final_map = False, n_gals = 100):
        """
        Produce volume map as 3D and 2D scatter plot of galaxies, if a catalog is provided.
        
        Arguments:
            bool final_map: flag to raise if the inference is finished
            int n_gals:     number of galaxies to plot
        """
        self.evaluate_volume_map()
        if self.catalog is None:
            return
            
        self.evaluate_catalog(final_map)
        
        # Cartesian plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(self.cat_to_plot_cartesian[:,0], self.cat_to_plot_cartesian[:,1], self.cat_to_plot_cartesian[:,2], c = self.p_cat_to_plot, marker = '.', alpha = 0.7, s = 0.5, cmap = 'Reds')
        vol_str = ['${0:.0f}\\%'.format(100*self.levels[-i])+ '\ \mathrm{CR}:'+'{0:.0f}'.format(self.volumes[-i]) + '\ \mathrm{Mpc}^3$' for i in range(len(self.volumes))]
        vol_str = '\n'.join(vol_str + ['${0}'.format(len(self.cat_to_plot_cartesian)) + '\ \mathrm{galaxies}\ \mathrm{in}\ '+'{0:.0f}\\%'.format(100*self.levels[np.where(self.levels == self.region)][0])+ '\ \mathrm{CR}$'])
        ax.text2D(0.05, 0.95, vol_str, transform=ax.transAxes)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        if final_map:
            fig.savefig(Path(self.volume_folder, self.name+'_cartesian_all.pdf'), bbox_inches = 'tight')
        else:
            fig.savefig(Path(self.volume_folder, self.name+'_cartesian_{0}'.format(self.n_pts)+'.pdf'), bbox_inches = 'tight')
        plt.close()
        
        # Celestial plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(self.cat_to_plot_celestial[:,0], self.cat_to_plot_celestial[:,1], self.cat_to_plot_celestial[:,2], c = self.p_cat_to_plot, marker = '.', alpha = 0.7, s = 0.5, cmap = 'Reds')
        vol_str = ['${0:.0f}\\%'.format(100*self.levels[-i])+ '\ \mathrm{CR}:'+'{0:.0f}'.format(self.volumes[-i]) + '\ \mathrm{Mpc}^3$' for i in range(len(self.volumes))]
        vol_str = '\n'.join(vol_str + ['${0}'.format(len(self.cat_to_plot_celestial)) + '\ \mathrm{galaxies}\ \mathrm{in}\ '+'{0:.0f}\\%'.format(100*self.levels[np.where(self.levels == self.region)][0])+ '\ \mathrm{CR}$'])
        ax.text2D(0.05, 0.95, vol_str, transform=ax.transAxes)
        ax.set_xlabel('$\\alpha$')
        ax.set_ylabel('$\\delta$')
        if final_map:
            fig.savefig(Path(self.volume_folder, self.name+'_all.pdf'), bbox_inches = 'tight')
            if self.next_plot < np.inf:
                fig.savefig(Path(self.gif_folder, '3d_'+self.name+'_all.png'), bbox_inches = 'tight')
        else:
            fig.savefig(Path(self.volume_folder, self.name+'_{0}'.format(self.n_pts)+'.pdf'), bbox_inches = 'tight')
            if self.next_plot < np.inf:
                fig.savefig(Path(self.gif_folder, '3d_'+self.name+'_{0}'.format(self.n_pts)+'.png'), bbox_inches = 'tight')
        plt.close()
        
        # 2D galaxy plot
        if self.virtual_observatory:
        # Limits for VO image
            fig_b = plt.figure()
            ax_b  = fig_b.add_subplot(111)
            c = ax_b.scatter(self.sorted_cat[:,0][:-int(n_gals):-1]*180./np.pi, self.sorted_cat[:,1][:-int(n_gals):-1]*180./np.pi, c = self.sorted_p_cat_to_plot[:-int(n_gals):-1], marker = '+', cmap = 'coolwarm', linewidths = 1)
            x_lim = ax_b.get_xlim()
            y_lim = ax_b.get_ylim()
            plt.close(fig_b)
            fig = plt.figure()
            # Download background
            if self.true_host is not None:
                pos = SkyCoord(self.true_host[0]*180./np.pi, self.true_host[1]*180./np.pi, unit = 'deg')
            else:
                pos = SkyCoord((x_lim[1]+x_lim[0])/2., (y_lim[1]+y_lim[0])/2., unit = 'deg')
            size = (Quantity(4, unit = 'deg'), Quantity(6, unit = 'deg'))
            ss = vo.regsearch(servicetype='image',waveband='optical', keywords=['SkyView'])[0]
            sia_results = ss.search(pos=pos, size=size, intersect='overlaps', format='image/fits')
            urls = [r.getdataurl() for r in sia_results]
            for attempt in range(10):
                # Download timeout
                try:
                    hdu = [fits.open(ff)[0] for ff in urls][0]
                except socket.timeout:
                    continue
                else:
                    break
            wcs = WCS(hdu.header)
            ax = fig.add_subplot(111, projection=wcs)
            ax.imshow(hdu.data,cmap = 'gray')
            ax.set_autoscale_on(False)
            c = ax.scatter(self.sorted_cat[:,0][:-int(n_gals):-1]*180./np.pi, self.sorted_cat[:,1][:-int(n_gals):-1]*180./np.pi, c = self.sorted_p_cat_to_plot[:-int(n_gals):-1], marker = '+', cmap = 'coolwarm', linewidths = 0.5, transform=ax.get_transform('world'), zorder = 100)
            c1 = ax.contour(self.ra_2d*180./np.pi, self.dec_2d*180./np.pi, self.log_p_skymap.T, np.sort(self.skymap_heights), colors = 'white', linewidths = 0.5, linestyles = 'solid', transform=ax.get_transform('world'), zorder = 99, alpha = 0)
            if self.true_host is not None:
                ax.scatter([self.true_host[0]*180./np.pi], [self.true_host[1]*180./np.pi], s=80, facecolors='none', edgecolors='g', label = '$\mathrm{' + self.host_name + '}$', transform=ax.get_transform('world'), zorder = 101)
            leg_col = 'white'
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            c = ax.scatter(self.sorted_cat[:,0][:-int(n_gals):-1], self.sorted_cat[:,1][:-int(n_gals):-1], c = self.sorted_p_cat_to_plot[:-int(n_gals):-1], marker = '+', cmap = 'coolwarm', linewidths = 1)
            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()
            c1 = ax.contour(self.ra_2d, self.dec_2d, self.log_p_skymap.T, np.sort(self.skymap_heights), colors = 'black', linewidths = 0.5, linestyles = 'solid')
            if self.true_host is not None:
                ax.scatter([self.true_host[0]], [self.true_host[1]], s=80, facecolors='none', edgecolors='g', label = '$\mathrm{' + self.host_name + '}$')
            leg_col = 'black'
        for i in range(len(self.areas)):
            c1.collections[i].set_label('${0:.0f}\\%'.format(100*self.levels[-i])+ '\ \mathrm{CR}:'+'{0:.1f}'.format(self.areas[-i]) + '\ \mathrm{deg}^2$')
        handles, labels = ax.get_legend_handles_labels()
        if self.n_gal_to_plot == -1 or self.n_gal_to_plot == len(self.catalog):
            lab_ngal = '${0}'.format(len(self.cat_to_plot_celestial)) + '\ \mathrm{galaxies}\ \mathrm{in}\ '+'{0:.0f}\\%'.format(100*self.levels[np.where(self.levels == self.region)][0])+ '\ \mathrm{CR}$'
        else:
            lab_ngal = '${0}'.format(len(self.cat_to_plot_celestial)) + '\ \mathrm{galaxies}\ \mathrm{in}\ '+'{0:.0f}\\%'.format(100*self.levels[np.where(self.levels == self.region)][0])+ '\ \mathrm{CR}$\n'+'$({0}'.format(self.n_gal_to_plot)+'\ \mathrm{shown})$'
        patch = mpatches.Patch(color='grey', label=lab_ngal, alpha = 0)
        handles.append(patch)
        plt.colorbar(c, label = '$p_{host}$')
        ax.set_xlabel('$\\alpha$')
        ax.set_ylabel('$\\delta$')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.legend(handles = handles, loc = 2, fontsize = 10, handlelength=0, labelcolor = leg_col)
        if final_map:
            fig.savefig(Path(self.skymap_folder, 'galaxies_'+self.name+'_all.pdf'), bbox_inches = 'tight')
        else:
            fig.savefig(Path(self.skymap_folder, 'galaxies_'+self.name+'_{0}'.format(self.n_pts)+'.pdf'), bbox_inches = 'tight')
        plt.close()
        
    def make_gif(self):
        """
        If the inference produced updated skymaps, join them into a single .gif file
        """
        files = [f for f in self.gif_folder.glob('3d_'+self.name + '*' + '.png')]
        if len(files) > 1:
            path_files = [str(f) for f in files]
            path_files.sort(key = natural_keys)
            images = []
            for file in path_files:
                images.append(imageio.imread(file))
            imageio.mimsave(Path(self.gif_folder, '3d_'+self.name + '.gif'), images, fps = 1)
        [f.unlink() for f in files]
        files = [f for f in self.gif_folder.glob(self.name + '*' + '.png')]
        if len(files) > 1:
            path_files = [str(f) for f in files]
            path_files.sort(key = natural_keys)
            images = []
            for file in path_files:
                images.append(imageio.imread(file))
            imageio.mimsave(Path(self.gif_folder, self.name + '.gif'), images, fps = 1)
        [f.unlink() for f in files]
    
    def make_entropy_plot(self):
        """
        Produce entropy plot and angular coefficient plot, if entropy = True
        """
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(self.R_S))*self.entropy_step, np.abs(self.R_S), color = 'steelblue', lw = 0.7)
        ax.set_ylabel('$S(N)\ [\mathrm{bits}]$')
        ax.set_xlabel('$N$')
        
        fig.savefig(Path(self.entropy_folder, self.name + '.pdf'), bbox_inches = 'tight')
        plt.close()

        fig, ax = plt.subplots()
        ax.axhline(0, lw = 0.5, ls = '--', color = 'r')
        ax.plot(np.arange(len(self.ac))*self.entropy_step + self.entropy_ac_step, self.ac, color = 'steelblue', lw = 0.7)
        ax.set_ylabel('$\\frac{dS(N)}{dN}$')
        ax.set_xlabel('$N$')
        
        fig.savefig(Path(self.entropy_folder, 'ang_coeff_'+self.name + '.pdf'), bbox_inches = 'tight')
        plt.close()
    
    def save_density(self):
        """
        Build and save density
        """
        density = self.build_mixture()
        with open(Path(self.density_folder, self.name + '_density.pkl'), 'wb') as dill_file:
            dill.dump(density, dill_file)
    
    def volume_N_plot(self):
        """
        Produce plot to show the evolution of credible area/volume with the number of samples
        """
        output = [self.N]
        header = 'N '
        
        fig, (ax_a, ax_v) = plt.subplots(2,1, sharex = True)
        
        for lev in self.levels:
            vol = self.volumes_N[lev]
            a   = self.areas_N[lev]
            output.append(a)
            output.append(vol)
            header = header + 'A_{0:.0f} V_{0:.0f} '.format(lev*100)
            
            ax_a.plot(self.N, a/a[-1], marker = 's', ls = '--', label = '${0:.0f}\%\ '.format(lev*100)+'\mathrm{CR}$')
            ax_v.plot(self.N, vol/vol[-1], marker = 's', ls = '--', label = '${0:.0f}\%\ '.format(lev*100)+'\mathrm{CR}$')
        ax_v.set_ylabel('$V/V_{f}$')
        ax_a.set_ylabel('$\Omega/\Omega_{f}$')
        ax_v.set_xlabel('$N_{\mathrm{samples}}$')
        ax_a.set_xscale('log')
        ax_v.set_yscale('log')
        ax_a.set_yscale('log')
        ax_a.legend(loc = 0, frameon = False, fontsize = 10)
        ax_v.legend(loc = 0, frameon = False, fontsize = 10)
        
        fig.savefig(Path(self.CR_folder, self.name + '.pdf'), bbox_inches = 'tight')
        np.savetxt(Path(self.CR_folder, self.name + '.txt'), np.array(output).T, header = header)
        plt.close()
        
    def density_from_samples(self, samples):
        """
        Reconstruct the probability density from a set of samples.
        Samples must be in celestial coordinate and in the following order: [ra, dec, dist].
        
        Arguments:
            iterable samples: set of volume samples
        """
        np.random.shuffle(samples)
        # Checking the posteriors are properly ordered:
        check_ra   = np.logical_and(samples[:,0] > 0, samples[:,0] < 2*np.pi).all()
        check_dec  = np.logical_and(samples[:,1] > -np.pi/2., samples[:,1] < np.pi/2.).all()
        check_dist = np.logical_and(samples[:,2] > 0, samples[:,2] < self.max_dist).all()
        if not (check_ra and check_dec and check_dist):
            raise FIGAROException("Samples are not in [RA, dec, DL] order or one or more points are outside the [[0, 2π], [-π/2, π/2], [0,{0:.0f}]] boundaries".format(self.max_dist))
        self.ac_cntr = self.n_sign_changes
        for i in tqdm(range(len(samples)), desc=self.name):
            self.add_new_point(samples[i])
            if self.entropy:
                if i%self.entropy_step == 0:
                    R_S = compute_entropy_single_draw(self, self.n_entropy_MC_draws)
                    self.R_S.append(R_S)
                    if self.n_pts//self.entropy_ac_step >= 1:
                        ac = angular_coefficient(self.N_for_ac + self.n_pts, self.R_S[-self.entropy_ac_step:])
                        if self.flag_skymap == False:
                            try:
                                if ac*self.ac[-1] < 0:
                                    self.ac_cntr = self.ac_cntr - 1
                            except IndexError: #Empty list
                                pass
                            if self.ac_cntr < 1:
                                self.flag_skymap = True
                                self.N.append(self.n_pts)
                                self.make_skymap()
                                self.make_volume_map(n_gals = self.n_gal_to_plot)
                                if self.next_plot < np.inf:
                                    self.next_plot = self.n_pts*2
                        self.ac.append(ac)
            else:
                self.flag_skymap = True
                        
        self.save_density()
        self.N.append(self.n_pts)
        self.plot_samples(self.n_pts, initial_samples = samples)
        self.make_skymap(final_map = True)
        self.make_volume_map(final_map = True, n_gals = self.n_gal_to_plot)
        if self.next_plot < np.inf:
            self.make_gif()
            self.volume_N_plot()
        if self.entropy:
            self.make_entropy_plot()
        if self.true_host is not None:
            self.compute_credible_levels_host()
