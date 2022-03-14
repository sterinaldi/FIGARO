import numpy as np
from itertools import product
import h5py
import re

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from corner import corner
import imageio

from scipy.special import logsumexp
import dill

from figaro.mixture import DPGMM
from figaro.transform import *
from figaro.coordinates import celestial_to_cartesian, cartesian_to_celestial, inv_Jacobian
from figaro.credible_regions import ConfidenceArea, ConfidenceVolume, FindNearest, FindLevelForHeight
from figaro.cosmology import CosmologicalParameters
from figaro.diagnostic import compute_entropy_rate_single_draw

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

# natural sorting.
# list.sort(key = natural_keys)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class VolumeReconstruction(DPGMM):
    def __init__(self, max_dist,
                       out_folder             = '.',
                       prior_pars             = None,
                       alpha0                 = 1,
                       n_gridpoints           = [720, 360, 100], # RA, dec, DL
                       name                   = 'skymap',
                       labels                 = ['$\\alpha$', '$\\delta$', '$D\ [Mpc]$'],
                       levels                 = [0.50, 0.90],
                       latex                  = False,
                       incr_plot              = False,
                       glade_file             = None,
                       cosmology              = {'h': 0.674, 'om': 0.315, 'ol': 0.685},
                       n_gal_to_print         = 100,
                       region_to_plot         = 0.5,
                       cat_bound              = 3000,
                       entropy_rate           = False,
                       true_host              = None,
                       entropy_rate_threshold = 1e-2,
                       entropy_rate_step      = 10,
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
        self.log_inv_J = -np.log(inv_Jacobian(self.grid)) - probit_logJ(self.probit_grid, self.bounds)
        self.inv_J = np.exp(self.log_inv_J)
        
        # True host
        self.true_host = true_host
        if self.true_host is not None:
            self.pixel_idx  = FindNearest(self.ra, self.dec, self.dist, self.true_host)
            self.true_pixel = np.array([self.ra[self.pixel_idx[0]], self.dec[self.pixel_idx[1]], self.dist[self.pixel_idx[2]]])
        
        # Credible regions levels
        self.levels      = np.array(levels)
        self.areas_N     = {cr:[] for cr in self.levels}
        self.volumes_N   = {cr:[] for cr in self.levels}
        self.N           = []
        self.flag_skymap = True
        if entropy_rate == True:
            self.flag_skymap = False
        
        # Catalog
        self.cosmology = CosmologicalParameters(cosmology['h'], cosmology['om'], cosmology['ol'], 1, 0)
        self.catalog   = None
        self.cat_bound = cat_bound
        if glade_file is not None and self.max_dist < self.cat_bound:
            self.catalog = self.load_glade(glade_file)
            self.cartesian_catalog = celestial_to_cartesian(self.catalog)
            self.probit_catalog    = transform_to_probit(self.cartesian_catalog, self.bounds)
            self.log_inv_J_cat     = -np.log(inv_Jacobian(self.catalog)) - probit_logJ(self.probit_catalog, self.bounds)
            self.inv_J_cat         = np.exp(self.log_inv_J)
        self.n_gal_to_print = n_gal_to_print
        if region_to_plot in self.levels:
            self.region = region_to_plot
        else:
            self.region = self.levels[0]
        
        # Entropy
        self.entropy_rate           = entropy_rate
        self.entropy_rate_step      = entropy_rate_step
        self.entropy_rate_threshold = entropy_rate_threshold
        self.R_S                    = []
        
        # Output
        self.name       = name
        self.labels     = labels
        self.out_folder = Path(out_folder).resolve()
        self.make_folders()
    
    def initialise(self, true_host = None):
        self.volume_already_evaluated = False
        super().initialise()
        self.true_host = true_host
        if self.true_host is not None:
            self.pixel_idx  = FindNearest(self.ra, self.dec, self.dist, self.true_host)
            self.true_pixel = np.array([self.ra[self.pixel_idx[0]], self.dec[self.pixel_idx[1]], self.dist[self.pixel_idx[2]]])
        
    def load_glade(self, glade_file):
        with h5py.File(glade_file, 'r') as f:
            ra  = np.array(f['ra'])
            dec = np.array(f['dec'])
            z   = np.array(f['z'])
        
        DL = self.cosmology.LuminosityDistance(z)
        catalog = np.array([ra, dec, DL]).T
        catalog = catalog[catalog[:,2] < self.max_dist]
        return catalog

    def make_folders(self):
        if not Path(self.out_folder, 'skymaps').exists():
            Path(self.out_folder, 'skymaps').mkdir()
        if not Path(self.out_folder, 'volume').exists():
            Path(self.out_folder, 'volume').mkdir()
        if not Path(self.out_folder, 'catalogs').exists():
            Path(self.out_folder, 'catalogs').mkdir()
        if not Path(self.out_folder, 'convergence').exists():
            Path(self.out_folder, 'convergence').mkdir()
        self.skymap_folder = Path(self.out_folder, 'skymaps', self.name)
        self.convergence_folder = Path(self.out_folder, 'convergence')
        if not self.convergence_folder.exists():
            self.convergence_folder.mkdir()
        if not self.skymap_folder.exists():
            self.skymap_folder.mkdir()
        if self.max_dist < self.cat_bound and self.catalog is not None:
            self.volume_folder = Path(self.out_folder, 'volume', self.name)
            if not self.volume_folder.exists():
                self.volume_folder.mkdir()
        self.catalog_folder = Path(self.out_folder, 'catalogs', self.name)
        if not self.catalog_folder.exists():
            self.catalog_folder.mkdir()
        self.density_folder = Path(self.out_folder, 'density')
        if not self.density_folder.exists():
            self.density_folder.mkdir()
        self.gif_folder = Path(self.out_folder, 'gif')
        if not self.gif_folder.exists():
            self.gif_folder.mkdir()
        self.entropy_rate_folder = Path(self.out_folder, 'entropy_rate')
        if not self.entropy_rate_folder.exists():
            self.entropy_rate_folder.mkdir()


    def add_sample(self, x):
        self.volume_already_evaluated = False
        cart_x = celestial_to_cartesian(x)
        self.add_new_point(cart_x)
        if self.flag_skymap and self.n_pts == self.next_plot:
            self.N.append(self.next_plot)
            self.next_plot = 2*self.next_plot
            self.make_skymap()
            self.make_volume_map(n_gals = self.n_gal_to_print)
    
    def sample_from_volume(self, n_samps):
        samples = self.sample_from_dpgmm(n_samps)
        return cartesian_to_celestial(samples)
    
    def plot_samples(self, n_samps, initial_samples = None):
        mix_samples = self.sample_from_volume(n_samps)
        if initial_samples is not None:
            c = corner(initial_samples, color = 'coral', labels = self.labels, truths = self.true_host, hist_kwargs={'density':True, 'label':'$\mathrm{Samples}$'})
            c = corner(mix_samples, fig = c, color = 'dodgerblue', labels = self.labels, hist_kwargs={'density':True, 'label':'$\mathrm{DPGMM}$'})
        else:
            c = corner(mix_samples, fig = c, color = 'dodgerblue', labels = self.labels, hist_kwargs={'density':True, 'label':'$\mathrm{DPGMM}$'})
        plt.legend(loc = 0, frameon = False,fontsize = 15, bbox_to_anchor = (1-0.05, 2.8))
        plt.savefig(Path(self.skymap_folder, 'corner_'+self.name+'.pdf'), bbox_inches = 'tight')
        plt.close()
    
    def evaluate_skymap(self):
        if not self.volume_already_evaluated:
            p_vol               = self._evaluate_mixture_in_probit(self.probit_grid) * self.inv_J
            self.norm_p_vol     = (p_vol*self.dD*self.dra*self.ddec).sum()
            self.log_norm_p_vol = np.log(self.norm_p_vol)
            self.p_vol          = p_vol/self.norm_p_vol
            
            # By default computes log(p_vol). If -infs are present, computes log_p_vol
            with np.errstate(divide='raise'):
                try:
                    self.log_p_vol = np.log(self.p_vol)
                except FloatingPointError:
                    self.log_p_vol = self._evaluate_log_mixture_in_probit(self.probit_grid) + self.log_inv_J - self.log_norm_p_vol
                    
            self.p_vol     = self.p_vol.reshape(len(self.ra), len(self.dec), len(self.dist))
            self.log_p_vol = self.log_p_vol.reshape(len(self.ra), len(self.dec), len(self.dist))
            self.volume_already_evaluated = True

        self.p_skymap = (self.p_vol*self.dD).sum(axis = -1)
        
        # By default computes log(p_skymap). If -infs are present, computes log_p_skymap
        with np.errstate(divide='raise'):
            try:
                self.log_p_skymap = np.log(self.p_skymap)
            except FloatingPointError:
                self.log_p_skymap = logsumexp(self.log_p_vol + np.log(self.dD), axis = -1)

        self.areas, self.skymap_idx_CR, self.skymap_heights = ConfidenceArea(self.log_p_skymap, self.ra, self.dec, adLevels = self.levels)
        for cr, area in zip(self.levels, self.areas):
            self.areas_N[cr].append(area)
    
    def evaluate_volume_map(self):
        if not self.volume_already_evaluated:
            p_vol               = self._evaluate_mixture_in_probit(self.probit_grid) * self.inv_J
            self.norm_p_vol     = (p_vol*self.dD*self.dra*self.ddec).sum()
            self.log_norm_p_vol = np.log(self.norm_p_vol)
            self.p_vol          = p_vol/self.norm_p_vol
            
            # By default, just uses log(p_vol). If -infs are present, computes log_p_vol
            with np.errstate(divide='raise'):
                try:
                    self.log_p_vol = np.log(self.p_vol)
                except FloatingPointError:
                    self.log_p_vol = self._evaluate_log_mixture_in_probit(self.probit_grid) + self.log_inv_J - self.log_norm_p_vol

            self.p_vol     = self.p_vol.reshape(len(self.ra), len(self.dec), len(self.dist))
            self.log_p_vol = self.log_p_vol.reshape(len(self.ra), len(self.dec), len(self.dist))
            self.volume_already_evaluated = True
            
        self.volumes, self.idx_CR, self.volume_heights = ConfidenceVolume(self.log_p_vol, self.ra, self.dec, self.dist, adLevels = self.levels)
        
        for cr, vol in zip(self.levels, self.volumes):
            self.volumes_N[cr].append(vol)
    
    def compute_credible_regions(self):
        self.log_p_vol_host    = self.log_p_vol[self.pixel_idx[0],self.pixel_idx[1],self.pixel_idx[2]]
        self.log_p_skymap_host = self.log_p_skymap[self.pixel_idx[0], self.pixel_idx[1]]
        
        self.CR_host           = FindLevelForHeight(self.log_p_skymap, self.log_p_skymap_host)
        self.CV_host           = FindLevelForHeight(self.log_p_vol, self.log_p_vol_host)
        
    
    def evaluate_catalog(self):
        log_p_cat                  = self._evaluate_log_mixture_in_probit(self.probit_catalog) + self.log_inv_J_cat - self.log_norm_p_vol
        self.log_p_cat_to_plot     = log_p_cat[np.where(log_p_cat > self.volume_heights[np.where(self.levels == self.region)])]
        self.p_cat_to_plot         = np.exp(self.log_p_cat_to_plot)
        self.cat_to_plot_celestial = self.catalog[np.where(log_p_cat > self.volume_heights[np.where(self.levels == self.region)])]
        self.cat_to_plot_cartesian = self.cartesian_catalog[np.where(log_p_cat > self.volume_heights[np.where(self.levels == self.region)])]
        
        self.sorted_cat = np.c_[self.cat_to_plot_celestial[np.argsort(self.log_p_cat_to_plot)], np.sort(self.log_p_cat_to_plot)][::-1]
        self.sorted_p_cat_to_plot = np.sort(self.p_cat_to_plot)[::-1]
        np.savetxt(Path(self.catalog_folder, self.name+'_{0}'.format(self.n_pts)+'.txt'), self.sorted_cat[:self.n_gal_to_print], header = 'ra dec dist logp')
    
    def make_skymap(self, final_map = False):
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
        ax.legend(handles = handles, loc = 0, frameon = False, fontsize = 10, handlelength=0, handletextpad=0, markerscale=0)
        if final_map:
            fig.savefig(Path(self.skymap_folder, self.name+'_all.pdf'), bbox_inches = 'tight')
            fig.savefig(Path(self.gif_folder, self.name+'_all.png'), bbox_inches = 'tight')
        else:
            fig.savefig(Path(self.skymap_folder, self.name+'_{0}'.format(self.n_pts)+'.pdf'), bbox_inches = 'tight')
            fig.savefig(Path(self.gif_folder, self.name+'_{0}'.format(self.n_pts)+'.png'), bbox_inches = 'tight')
        plt.close()
    
    def make_volume_map(self, final_map = False, n_gals = 100):
        self.evaluate_volume_map()
        if self.catalog is None:
            return
            
        self.evaluate_catalog()
        
        # Cartesian plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(self.cat_to_plot_cartesian[:,0], self.cat_to_plot_cartesian[:,1], self.cat_to_plot_cartesian[:,2], c = self.p_cat_to_plot, marker = '.', alpha = 0.7, s = 0.5, cmap = 'Reds')
        vol_str = ['${0:.0f}\\%'.format(100*self.levels[-i])+ '\ \mathrm{CR}:'+'{0:.0f}'.format(self.volumes[-i]) + '\ \mathrm{Mpc}^3$' for i in range(len(self.volumes))]
        vol_str = '\n'.join(vol_str + ['$N_{\mathrm{gal}}\ \mathrm{in}\ '+'{0:.0f}\\%'.format(100*self.levels[np.where(self.levels == self.region)][0])+ '\ \mathrm{CR}:'+'{0}$'.format(len(self.cat_to_plot_cartesian))])
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
        vol_str = '\n'.join(vol_str + ['$\mathrm{Galaxies}\ \mathrm{in}\ '+'{0:.0f}\\%'.format(100*self.levels[np.where(self.levels == self.region)][0])+ '\ \mathrm{CR}:'+'{0}$'.format(len(self.cat_to_plot_celestial))])
        ax.text2D(0.05, 0.95, vol_str, transform=ax.transAxes)
        ax.set_xlabel('$\\alpha$')
        ax.set_ylabel('$\\delta$')
        if final_map:
            fig.savefig(Path(self.volume_folder, self.name+'_all.pdf'), bbox_inches = 'tight')
            fig.savefig(Path(self.gif_folder, '3d_'+self.name+'_all.png'), bbox_inches = 'tight')
        else:
            fig.savefig(Path(self.volume_folder, self.name+'_{0}'.format(self.n_pts)+'.pdf'), bbox_inches = 'tight')
            fig.savefig(Path(self.gif_folder, '3d_'+self.name+'_{0}'.format(self.n_pts)+'.png'), bbox_inches = 'tight')
        plt.close()
        
        # 2D galaxy plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        c = ax.scatter(self.sorted_cat[:,0][:-int(n_gals):-1], self.sorted_cat[:,1][:-int(n_gals):-1], c = self.sorted_p_cat_to_plot[:-int(n_gals):-1], marker = '+', cmap = 'coolwarm', linewidths= 1)
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        c1 = ax.contour(self.ra_2d, self.dec_2d, self.log_p_skymap.T, np.sort(self.skymap_heights), colors = 'black', linewidths = 0.5, linestyles = 'solid')
        if self.true_host is not None:
            ax.scatter([self.true_pixel[0]], [self.true_pixel[1]], s=80, facecolors='none', edgecolors='g', label = '$\mathrm{Host}$')
        for i in range(len(self.areas)):
            c1.collections[i].set_label('${0:.0f}\\%'.format(100*self.levels[-i])+ '\ \mathrm{CR}:'+'{0:.1f}'.format(self.areas[-i]) + '\ \mathrm{deg}^2$')
        handles, labels = ax.get_legend_handles_labels()
        patch = mpatches.Patch(color='grey', label='${0}'.format(len(self.cat_to_plot_celestial))+'\ \mathrm{galaxies}$', alpha = 0)
        handles.append(patch)
        ax.set_xlabel('$\\alpha$')
        ax.set_ylabel('$\\delta$')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.legend(handles = handles, loc = 0, frameon = False, fontsize = 10, handlelength=0)
        if final_map:
            fig.savefig(Path(self.skymap_folder, 'galaxies_'+self.name+'_all.pdf'), bbox_inches = 'tight')
        else:
            fig.savefig(Path(self.skymap_folder, 'galaxies_'+self.name+'_{0}'.format(self.n_pts)+'.pdf'), bbox_inches = 'tight')
        plt.close()
        
    def make_gif(self):
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
        fig, ax = plt.subplots()
        ax.axhline(self.entropy_rate_threshold, lw = 0.5, ls = '--', color = 'r')
        ax.plot(np.arange(len(self.R_S))*self.entropy_rate_step, self.R_S, color = 'steelblue', lw = 0.7)
        ax.set_ylabel('$R_S(N)\ [\mathrm{bits/sample}]$')
        ax.set_xlabel('$N$')
        
        fig.savefig(Path(self.entropy_rate_folder, self.name + '.pdf'), bbox_inches = 'tight')
    
    def save_density(self):
        density = self.build_mixture()
        with open(Path(self.density_folder, self.name + '_density.pkl'), 'wb') as dill_file:
            dill.dump(density, dill_file)
    
    def volume_N_plot(self):
        
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
        
        fig.savefig(Path(self.convergence_folder, self.name + '.pdf'), bbox_inches = 'tight')
        np.savetxt(Path(self.convergence_folder, self.name + '.txt'), np.array(output).T, header = header)
        
        plt.close()
        
    def density_from_samples(self, samples):
        for i in tqdm(range(len(samples)), desc=self.name):
            self.add_sample(samples[i])
            if self.entropy_rate:
                if i%self.entropy_rate_step == 0:
                    R_S = compute_entropy_rate_single_draw(self)
                    self.R_S.append(R_S)
                    if np.abs(R_S) < self.entropy_rate_threshold:
                        self.flag_skymap = True
                        self.N.append(self.n_pts)
                        self.make_skymap()
                        self.make_volume_map(n_gals = self.n_gal_to_print)
                        if self.next_plot < np.inf:
                            self.next_plot = self.n_pts*2
        
        self.save_density()
        self.N.append(self.n_pts)
        self.plot_samples(self.n_pts, initial_samples = samples)
        self.make_skymap(final_map = True)
        self.make_volume_map(final_map = True, n_gals = self.n_gal_to_print)
        self.make_gif()
        self.volume_N_plot()
        if self.entropy_rate:
            self.make_entropy_plot()
        if self.true_host is not None:
            self.compute_credible_regions()
