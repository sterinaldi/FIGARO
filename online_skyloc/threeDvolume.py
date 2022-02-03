import numpy as np
from itertools import product
import h5py

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from corner import corner

import dill

from online_skyloc.incremental_gibbs import mixture
from online_skyloc.transform import *
from online_skyloc.coordinates import celestial_to_cartesian, cartesian_to_celestial, inv_Jacobian
from online_skyloc.credible_regions import ConfidenceArea, ConfidenceVolume
from online_skyloc.cosmology import CosmologicalParameters

from pathlib import Path
from distutils.spawn import find_executable
from tqdm import tqdm

rcParams["xtick.labelsize"]=14
rcParams["ytick.labelsize"]=14
rcParams["xtick.direction"]="in"
rcParams["ytick.direction"]="in"
rcParams["legend.fontsize"]=15
rcParams["axes.labelsize"]=16
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.6

class VolumeReconstruction(mixture):
    def __init__(self, max_dist,
                       out_folder   = '.',
                       prior_pars   = None,
                       alpha0       = 1,
                       sigma_max    = 0.05,
                       n_gridpoints = [720, 360, 100], # RA, dec, DL
                       name         = 'skymap',
                       labels       = ['$\\alpha$', '$\\delta$', '$D\ [Mpc]$'],
                       levels       = [0.50, 0.90],
                       latex        = False,
                       incr_plot    = False,
                       glade_file   = None,
                       cosmology    = {'h': 0.674, 'om': 0.315, 'ol': 0.685},
                       n_gal_to_print = 100,
                       ):
                
        self.max_dist = max_dist
        bounds = np.array([[-max_dist, max_dist] for _ in range(3)])
        self.volume_already_evaluated = False
        
        super().__init__(bounds, prior_pars, alpha0, sigma_max)
        
        if incr_plot:
            self.next_plot = 20
        else:
            self.next_plot = np.inf

        if latex:
            if find_executable('latex'):
                rcParams["text.usetex"] = True
        
        # Grid
        self.ra   = np.linspace(0,2*np.pi, n_gridpoints[0])
        self.dec  = np.linspace(-np.pi/2*0.99, np.pi/2.*0.99, n_gridpoints[1])
        self.dist = np.linspace(max_dist*0.01, max_dist*0.99, n_gridpoints[2])
        self.dD   = np.diff(self.dist)[0]
        self.dra  = np.diff(self.ra)[0]
        self.ddec = np.diff(self.dec)[0]
        self.grid = np.array([np.array(v) for v in product(*(self.ra,self.dec,self.dist))])
        self.grid2d = np.array([np.array(v) for v in product(*(self.ra,self.dec))])
        self.ra_2d, self.dec_2d = np.meshgrid(self.ra, self.dec)
        self.cartesian_grid = celestial_to_cartesian(self.grid)
        self.probit_grid = transform_to_probit(self.cartesian_grid, self.bounds)

        # Jacobians
        self.log_inv_J = -np.log(inv_Jacobian(self.grid)).reshape(len(self.ra), len(self.dec), len(self.dist)) + probit_logJ(self.probit_grid, self.bounds).reshape(len(self.ra), len(self.dec), len(self.dist))
        self.inv_J = np.exp(self.log_inv_J)
        
        # Credible regions levels
        self.levels = levels
        
        # Output
        self.out_folder = Path(out_folder).resolve()
        if not Path(out_folder, 'skymaps').exists():
            Path(out_folder, 'skymaps').mkdir()
        if not Path(out_folder, 'volume').exists():
            Path(out_folder, 'volume').mkdir()
        if not Path(out_folder, 'catalogs').exists():
            Path(out_folder, 'catalogs').mkdir()
        self.skymap_folder = Path(out_folder, 'skymaps', name)
        if not self.skymap_folder.exists():
            self.skymap_folder.mkdir()
        self.volume_folder = Path(out_folder, 'volume', name)
        if not self.volume_folder.exists():
            self.volume_folder.mkdir()
        self.catalog_folder = Path(out_folder, 'catalogs', name)
        if not self.catalog_folder.exists():
            self.catalog_folder.mkdir()
        self.density_folder = Path(out_folder, 'density')
        if not self.density_folder.exists():
            self.density_folder.mkdir()
        self.name = name
        self.labels = labels
        
        # Catalog
        self.cosmology = CosmologicalParameters(cosmology['h'], cosmology['om'], cosmology['ol'], 1, 0)
        self.catalog   = None
        if glade_file is not None:
            self.catalog = self.load_glade(glade_file)
            self.cartesian_catalog = celestial_to_cartesian(self.catalog)
            self.probit_catalog    = transform_to_probit(self.cartesian_catalog, self.bounds)
            self.log_inv_J_cat     = -np.log(inv_Jacobian(self.catalog)) + probit_logJ(self.probit_catalog, self.bounds)
            self.inv_J_cat         = np.exp(self.log_inv_J)
        self.n_gal_to_print = n_gal_to_print

    def load_glade(self, glade_file):
        with h5py.File(glade_file, 'r') as f:
            ra  = np.array(f['ra'])
            dec = np.array(f['dec'])
            z   = np.array(f['z'])
        
        DL = self.cosmology.LuminosityDistance(z)
        catalog = np.array([ra, dec, DL]).T
        catalog = catalog[catalog[:,2] < self.max_dist]
        return catalog

    def add_sample(self, x):
        self.volume_already_evaluated = False
        cart_x = celestial_to_cartesian(x)
        self.add_new_point(cart_x)
        if self.n_pts == self.next_plot:
            self.next_plot = 2*self.next_plot
            self.make_skymap()
            self.make_volume_map()
    
    def sample_from_volume(self, n_samps):
        samples = self.sample_from_dpgmm(n_samps)
        return cartesian_to_celestial(samples)
    
    def plot_samples(self, n_samps, initial_samples = None):
        mix_samples = self.sample_from_volume(n_samps)
        if initial_samples is not None:
            c = corner(initial_samples, color = 'coral', labels = self.labels, hist_kwargs={'density':True, 'label':'$\mathrm{Samples}$'})
            c = corner(mix_samples, fig = c, color = 'dodgerblue', labels = self.labels, hist_kwargs={'density':True, 'label':'$\mathrm{DPGMM}$'})
        else:
            c = corner(mix_samples, fig = c, color = 'dodgerblue', labels = self.labels, hist_kwargs={'density':True, 'label':'$\mathrm{DPGMM}$'})
        plt.legend(loc = 0, frameon = False,fontsize = 15, bbox_to_anchor = (1-0.05, 2.8))
        plt.savefig(Path(self.skymap_folder, 'corner_'+self.name+'.pdf'), bbox_inches = 'tight')
        plt.close()
    
    def evaluate_skymap(self):
        if not self.volume_already_evaluated:
            p_vol               = self._evaluate_mixture_in_probit(self.probit_grid).reshape(len(self.ra), len(self.dec), len(self.dist)) * self.inv_J
            self.norm_p_vol     = (p_vol*self.dD*self.dra*self.ddec).sum()
            self.log_norm_p_vol = np.log(self.norm_p_vol)
            self.p_vol          = p_vol/self.norm_p_vol
            self.log_p_vol      = np.log(self.p_vol)
            self.volume_already_evaluated = True
        self.p_skymap = (p_vol*self.dD).sum(axis = -1)
        self.log_p_skymap = np.log(self.p_skymap)
        self.areas, self.skymap_idx_CR, self.skymap_heights = ConfidenceArea(self.log_p_skymap, self.ra, self.dec, adLevels = self.levels)
    
    def evaluate_volume_map(self):
        if not self.volume_already_evaluated:
            p_vol               = self._evaluate_mixture_in_probit(self.probit_grid).reshape(len(self.ra), len(self.dec), len(self.dist)) * self.inv_J
            self.norm_p_vol     = (p_vol*self.dD*self.dra*self.ddec).sum()
            self.log_norm_p_vol = np.log(self.norm_p_vol)
            self.p_vol          = p_vol/self.norm_p_vol
            self.log_p_vol      = np.log(self.p_vol)
            self.volume_already_evaluated = True
        self.volumes, self.idx_CR, self.volume_heights = ConfidenceVolume(self.log_p_vol, self.ra, self.dec, self.dist, adLevels = self.levels)
        
    def evaluate_catalog(self):
        log_p_cat          = self._evaluate_log_mixture_in_probit(self.probit_catalog) + self.log_inv_J_cat - self.log_norm_p_vol
        self.log_p_cat_to_plot = log_p_cat[np.where(log_p_cat > self.volume_heights[0])]
        self.p_cat_to_plot     = np.exp(self.log_p_cat_to_plot)
        self.cat_to_plot_celestial = self.catalog[np.where(log_p_cat > self.volume_heights[0])]
        self.cat_to_plot_cartesian = self.cartesian_catalog[np.where(log_p_cat > self.volume_heights[0])]
        
        sorted_cat = np.c_[self.cat_to_plot_celestial[np.argsort(self.log_p_cat_to_plot)], np.sort(self.log_p_cat_to_plot)][::-1]
        np.savetxt(Path(self.catalog_folder, self.name+'_{0}'.format(self.n_pts)+'.txt'), sorted_cat, header = 'ra dec dist logp')
    
    def make_skymap(self, final_map = False):
        self.evaluate_skymap()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        c = ax.contourf(self.ra_2d, self.dec_2d, self.p_skymap.T, 990, cmap = 'Reds')
        c1 = ax.contour(self.ra_2d, self.dec_2d, self.log_p_skymap.T, np.sort(self.skymap_heights), colors = 'black', linewidths = 0.5, linestyles = 'dashed')
        ax.clabel(c1, fmt = {l:'{0:.0f}%'.format(100*s) for l,s in zip(c1.levels, self.levels[::-1])}, fontsize = 5)
        for i in range(len(self.areas)):
            c1.collections[i].set_label('${0:.0f}\\%'.format(100*self.levels[-i])+ '\ \mathrm{CR}:'+'{0:.1f}'.format(self.areas[-i]) + '\ \mathrm{deg}^2$')
        handles, labels = ax.get_legend_handles_labels()
        patch = mpatches.Patch(color='grey', label='$N_{\mathrm{samples}} =' +'{0}$'.format(self.n_pts), alpha = 0)
        handles.append(patch)
        ax.set_xlabel('$\\alpha$')
        ax.set_ylabel('$\\delta$')
        ax.legend(handles = handles, loc = 0, frameon = False, fontsize = 10, handlelength=0, handletextpad=0, markerscale=0)
        if final_map:
            fig.savefig(Path(self.skymap_folder, self.name+'_all.pdf'), bbox_inches = 'tight')
        else:
            fig.savefig(Path(self.skymap_folder, self.name+'_{0}'.format(self.n_pts)+'.pdf'), bbox_inches = 'tight')
        plt.close()
    
    def make_volume_map(self, final_map = False):
        self.evaluate_volume_map()
        self.evaluate_catalog()
        
        # Cartesian plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(self.cat_to_plot_cartesian[:,0], self.cat_to_plot_cartesian[:,1], self.cat_to_plot_cartesian[:,2], c = self.p_cat_to_plot, marker = '.', alpha = 0.5, s = 0.5)
        vol_str = ['${0:.0f}\\%'.format(100*self.levels[-i])+ '\ \mathrm{CR}:'+'{0:.0f}'.format(self.volumes[-i]) + '\ \mathrm{Mpc}^3$' for i in range(len(self.volumes))]
        vol_str = '\n'.join(vol_str + ['$N_{\mathrm{gal}}\ \mathrm{in}\ '+'{0:.0f}\\%'.format(100*self.levels[0])+ '\ \mathrm{CR}:'+'{0}$'.format(len(self.cat_to_plot_cartesian))])
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
        ax.scatter(self.cat_to_plot_celestial[:,0], self.cat_to_plot_celestial[:,1], self.cat_to_plot_celestial[:,2], c = self.p_cat_to_plot, marker = '.', alpha = 0.5, s = 0.5)
        vol_str = ['${0:.0f}\\%'.format(100*self.levels[-i])+ '\ \mathrm{CR}:'+'{0:.0f}'.format(self.volumes[-i]) + '\ \mathrm{Mpc}^3$' for i in range(len(self.volumes))]
        vol_str = '\n'.join(vol_str + ['$N_{\mathrm{gal}}\ \mathrm{in}\ '+'{0:.0f}\\%'.format(100*self.levels[0])+ '\ \mathrm{CR}:'+'{0}$'.format(len(self.cat_to_plot_celestial))])
        ax.text2D(0.05, 0.95, vol_str, transform=ax.transAxes)
        ax.set_xlabel('$\\alpha$')
        ax.set_ylabel('$\\delta$')
        ax.set_zlabel('$D_L\ [\mathrm{Mpc}]$')
        if final_map:
            fig.savefig(Path(self.volume_folder, self.name+'_all.pdf'), bbox_inches = 'tight')
        else:
            fig.savefig(Path(self.volume_folder, self.name+'_{0}'.format(self.n_pts)+'.pdf'), bbox_inches = 'tight')
        plt.close()
    
    def save_density(self):
        with open(Path(self.density_folder, self.name + '_density.pkl'), 'wb') as dill_file:
            dill.dump(self, dill_file)

    def density_from_samples(self, samples):
        n_samps = len(samples)
        samples_copy = np.copy(samples)
        for s in tqdm(samples_copy):
            self.add_sample(s)
        self.plot_samples(n_samps, initial_samples = samples)
        self.make_skymap(final_map = True)
        if self.catalog is not None:
            self.make_volume_map(final_map = True)
        self.save_density()

