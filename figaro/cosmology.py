import numpy as np
#from scipy.optimize import newton
from numba import njit
from scipy.interpolate import interp1d
from astropy.cosmology import wCDM, z_at_value
from astropy.units import Quantity
#from lal._lal import LuminosityDistance, ComovingTransverseDistance, ComovingLOSDistance, HubbleDistance, HubbleParameter, UniformComovingVolumeDensity, ComovingVolumeElement, ComovingVolume, CreateCosmologicalParameters

class CosmologicalParameters:
    """
    Wrapper for Astropy functions in a single class.
    
    Arguments:
        double h:   normalised hubble constant h = H0/100 km/Mpc/s
        double om:  matter energy density
        double ol:  cosmological constant density
        double w0:  0th order dark energy equation of state parameter
        double w1:  1st order dark energy equation of state parameter
        double w2:  2nd order dark energy equation of state parameter
        
    Returns:
        CosmologicalParameters: instance of CosmologicalParameters class
    """
    def __init__(self, h, om, ol, w0, w1, w2):
        self.h = h
        self.om = om
        self.ol = ol
        self.ok = 1.-om-ol
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.Cosmology = wCDM(H0 = self.h*100, Om0 = self.om, Ode0 = self.ol, w0 = self.w0)#, self.w1, self.w2)
#        self.HubbleDistance = HubbleDistance(self._CosmologicalParameters)
        
#    def _vectorise(func):
#        def vectorised_func(self, x):
#            if hasattr(x, "__iter__"):
#                return np.array([func(self, xi) for xi in x])
#            else:
#                return func(self, x)
#        return vectorised_func
#    
#    @_vectorise
    def HubbleParameter(self, z):
        return self.Cosmology.H(z).value

#    @_vectorise
    def LuminosityDistance(self, z):
        return self.cosmology.luminosity_distance(z).value
    
#    @_vectorise
    def ComovingTransverseDistance(self, z):
        return self.Cosmology.comoving_transverse_distance(z).value
     
#    @_vectorise
    def ComovingLOSDistance(self, z):
        return self.Cosmology.comoving_distance(z).value
     
#    @_vectorise
#    def UniformComovingVolumeDensity(self, z):
#        return UniformComovingVolumeDensity(z, self._CosmologicalParameters)
    
    #@_vectorise
    def ComovingVolumeElement(self, z):
        return self.Cosmology.differential_comoving_volume(z).value
    
    #@_vectorise
    def ComovingVolume(self, z):
        return self.Cosmology.comoving_volume(z).value
    
#    @_vectorise
#    def dDTdDC(self, DC):
#        if self.ok == 0.:
#            return 1.
#        elif self.ok > 0.:
#            return np.cosh(np.sqrt(self.ok)*DC/self.HubbleDistance)
#        else:
#            return np.cos(np.sqrt(-self.ok)*DC/self.HubbleDistance)
    
#    @_vectorise
#    def dDLdz(self, z):
#        DC   = self.ComovingLOSDistance(z)
#        DT   = self.ComovingTransverseDistance(z)
#        invE = self.HubbleParameter(z)
#        return DT + (1.+z)*self.dDTdDC(DC)*self.HubbleDistance*invE
    
#    @_vectorise
    def Redshift(self, DL):
        if DL == 0.:
            return 0.
        else:
            return z_at_value(self.Cosmology.luminosity_distance, Quantity(DL, 'Mpc'))

Planck18 = CosmologicalParameters(0.674, 0.315, 0.685, -1)#, 0, 0)
Planck15 = CosmologicalParameters(0.679, 0.3065, 0.6935, -1)#, 0, 0)

# Interpolants up to z = 2.5
z = np.linspace(0,2.5,1000)
dvdz_planck18 = Planck18.ComovingVolumeElement(z)/1e9 # In Gpc
dvdz_planck15 = Planck15.ComovingVolumeElement(z)/1e9 # In Gpc

@njit
def dVdz_approx_planck15(x):
    return np.interp(x, z, dvdz_planck15)

@njit
def dVdz_approx_planck18(x):
    return np.interp(x, z, dvdz_planck18)

def _decorator_dVdz(func, approx, z_index, z_max):
    reg_const = (1+z_max)/approx(z_max)
    def decorated_func(x):
        return func(x)*approx(x[:,z_index])/(1+x[:,z_index]) * reg_const
    return decorated_func
