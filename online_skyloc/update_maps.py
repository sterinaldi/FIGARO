import numpy as np
import ray

@ray.remote
class volume:
    """
    class producing the volume reconstruction, sky maps
    and galaxy probability (if available)
    """
    def __init__(self, galaxy_catalog = None):
        self.galaxy_catalog = galaxy_catalog
        self.samples        = None
        
    def run(self, sampler):
        """
        Continously running process, queries the sampler
        for updates and launches the post-processing
        when it receives samples
        """
        
        while True:
            
            samples = ray.get(sampler)
            if samples == "Done":
                break
            
            self.update_maps(samples)
    
    def update_samples(self, samples):
        if self.samples is None:
            self.samples = samples
        else:
            self.samples = np.column_stack((self.samples,samples))
    
    def update_maps(self, samples):
    
        self.update_samples(samples)



        
    
        def rank_galaxies(self):
        
        sys.stderr.write("Ranking the galaxies: computing log posterior for %d galaxies\n"%(self.catalog.shape[0]))
        jobs        = ((self.density,np.array((d,dec,ra))) for d, dec, ra in zip(self.catalog[:,2],self.catalog[:,1],self.catalog[:,0]))
        results     = self.pool.imap(logPosterior, jobs, chunksize = np.int(self.catalog.shape[0]/ (self.nthreads * 16)))
        logProbs    = np.array([r for r in results])

        idx         = ~np.isnan(logProbs)
        self.ranked_probability = logProbs[idx]
        self.ranked_ra          = self.catalog[idx,0]
        self.ranked_dec         = self.catalog[idx,1]
        self.ranked_dl          = self.catalog[idx,2]
        self.ranked_zs          = self.catalog[idx,3]
        self.ranked_zp          = self.catalog[idx,4]
        
        order                   = self.ranked_probability.argsort()[::-1]
        
        self.ranked_probability = self.ranked_probability[order]
        self.ranked_ra          = self.ranked_ra[order]
        self.ranked_dec         = self.ranked_dec[order]
        self.ranked_dl          = self.ranked_dl[order]
        self.ranked_zs          = self.ranked_zs[order]
        self.ranked_zp          = self.ranked_zp[order]
    
    def evaluate_volume_map(self):
        N = self.bins[0]*self.bins[1]*self.bins[2]
        sys.stderr.write("computing log posterior for %d grid points\n"%N)
        sample_args         = ((self.density,np.array((d,dec,ra))) for d in self.grid[0] for dec in self.grid[1] for ra in self.grid[2])
        results             = self.pool.imap(logPosterior, sample_args, chunksize = N//(self.nthreads * 32))
        self.log_volume_map = np.array([r for r in results]).reshape(self.bins[0],self.bins[1],self.bins[2])
        self.volume_map     = np.exp(self.log_volume_map)
        # normalise
        dsquared         = self.grid[0]**2
        cosdec           = np.cos(self.grid[1])
        self.volume_map /= np.sum(self.volume_map*dsquared[:,None,None]*cosdec[None,:,None]*self.dD*self.dRA*self.dDEC)

    def evaluate_sky_map(self):
        dsquared        = self.grid[0]**2
        self.skymap     = np.trapz(dsquared[:,None,None]*self.volume_map, x=self.grid[0], axis=0)
        self.log_skymap = np.log(self.skymap)
    
    def evaluate_distance_map(self):
        cosdec                  = np.cos(self.grid[1])
        intermediate            = np.trapz(self.volume_map, x=self.grid[2], axis=2)
        self.distance_map       = np.trapz(cosdec*intermediate, x=self.grid[1], axis=1)
        self.log_distance_map   = np.log(self.distance_map)
        self.distance_map      /= (self.distance_map*np.diff(self.grid[0])[0]).sum()

    def ConfidenceVolume(self, adLevels):
        # create a normalized cumulative distribution
        self.log_volume_map_sorted  = np.sort(self.log_volume_map.flatten())[::-1]
        self.log_volume_map_cum     = fast_log_cumulative(self.log_volume_map_sorted)
        
        # find the indeces  corresponding to the given CLs
        adLevels        = np.ravel([adLevels])
        args            = [(self.log_volume_map_sorted,self.log_volume_map_cum,level) for level in adLevels]
        adHeights       = self.pool.map(FindHeights,args)
        self.heights    = {str(lev):hei for lev,hei in zip(adLevels,adHeights)}
        volumes         = []
        
        for height in adHeights:
            
            (index_d, index_dec, index_ra,) = np.where(self.log_volume_map>=height)
            volumes.append(np.sum([self.grid[0][i_d]**2. *np.cos(self.grid[1][i_dec]) * self.dD * self.dRA * self.dDEC for i_d,i_dec in zip(index_d,index_dec)]))

        self.volume_confidence = np.array(volumes)

        if self.injection is not None:
            ra,dec           = self.injection.get_ra_dec()
            distance         = self.injection.distance
            logPval          = logPosterior((self.density,np.array((distance,dec,ra))))
            confidence_level = np.exp(self.log_volume_map_cum[np.abs(self.log_volume_map_sorted-logPval).argmin()])
            height           = FindHeights((self.log_volume_map_sorted,self.log_volume_map_cum,confidence_level))
            (index_d, index_dec, index_ra,) = np.where(self.log_volume_map>=height)
            searched_volume  = np.sum([self.grid[0][i_d]**2. *np.cos(self.grid[1][i_dec]) * self.dD * self.dRA * self.dDEC for i_d,i_dec in zip(index_d,index_dec)])
            self.injection_volume_confidence    = confidence_level
            self.injection_volume_height        = height
            return self.volume_confidence,(confidence_level,searched_volume)

        del self.log_volume_map_sorted
        del self.log_volume_map_cum
        return self.volume_confidence,None

    def ConfidenceArea(self, adLevels):
        
        # create a normalized cumulative distribution
        self.log_skymap_sorted  = np.sort(self.log_skymap.flatten())[::-1]
        self.log_skymap_cum     = fast_log_cumulative(self.log_skymap_sorted)
        # find the indeces  corresponding to the given CLs
        adLevels                = np.ravel([adLevels])
        args                    = [(self.log_skymap_sorted,self.log_skymap_cum,level) for level in adLevels]
        adHeights               = self.pool.map(FindHeights,args)
        areas                   = []
        
        for height in adHeights:
            (index_dec,index_ra,) = np.where(self.log_skymap>=height)
            areas.append(np.sum([self.dRA*np.cos(self.grid[1][i_dec])*self.dDEC for i_dec in index_dec])*(180.0/np.pi)**2.0)
    
        self.area_confidence = np.array(areas)
        
        if self.injection is not None:
            ra,dec                  = self.injection.get_ra_dec()
            id_ra                   = np.abs(self.grid[2]-ra).argmin()
            id_dec                  = np.abs(self.grid[1]-dec).argmin()
            logPval                 = self.log_skymap[id_dec,id_ra]
            confidence_level        = np.exp(self.log_skymap_cum[np.abs(self.log_skymap_sorted-logPval).argmin()])
            height                  = FindHeights((self.log_skymap_sorted,self.log_skymap_cum,confidence_level))
            (index_dec,index_ra,)   = np.where(self.log_skymap >= height)
            searched_area           = np.sum([self.dRA*np.cos(self.grid[1][i_dec])*self.dDEC for i_dec in index_dec])*(180.0/np.pi)**2.0
            
            return self.area_confidence,(confidence_level,searched_area)

        del self.log_skymap_sorted
        del self.log_skymap_cum
        return self.area_confidence,None

    def ConfidenceDistance(self, adLevels):
        cumulative_distribution     = np.cumsum(self.distance_map*self.dD)
        distances                   = []
        
        for cl in adLevels:
            idx = np.abs(cumulative_distribution-cl).argmin()
            distances.append(self.grid[0][idx])
        
        self.distance_confidence = np.array(distances)

        if self.injection!=None:
            idx                 = np.abs(self.injection.distance-self.grid[0]).argmin()
            confidence_level    = cumulative_distribution[idx]
            searched_distance   = self.grid[0][idx]
            return self.distance_confidence,(confidence_level,searched_distance)

        return self.distance_confidence,None
            




