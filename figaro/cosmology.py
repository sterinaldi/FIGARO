import numpy as np
from numba import njit
from scipy.interpolate import interp1d
from scipy.optimize import newton
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio") # Silence LAL warnings with ipython
from astropy.cosmology import wCDM, z_at_value
from astropy.units import Quantity
try:
    from lal._lal import LuminosityDistance, ComovingTransverseDistance, ComovingLOSDistance, HubbleDistance, HubbleParameter, UniformComovingVolumeDensity, ComovingVolumeElement, ComovingVolume, CreateCosmologicalParameters
    use_lal = True
except :
    use_lal = False

class CosmologicalParameters:
    def __new__(self, *args, **kwargs):
        if use_lal:
            return CosmologicalParameters_lal(*args, **kwargs)
        else:
            return CosmologicalParameters_astropy(*args, **kwargs)

class CosmologicalParameters_lal:
    """
    Wrapper for LAL functions in a single class.
    
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
    def __init__(self, h, om, ol, w0, w1 = 0, w2 = 0):
        self.h = h
        self.om = om
        self.ol = ol
        self.ok = 1.-om-ol
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self._CosmologicalParameters = CreateCosmologicalParameters(self.h, self.om, self.ol, self.w0, self.w1, self.w2)
        self.HubbleDistance = HubbleDistance(self._CosmologicalParameters)
        
    def _vectorise(func):
        def vectorised_func(self, x):
            if hasattr(x, "__iter__"):
                return np.array([func(self, xi) for xi in x])
            else:
                return func(self, x)
        return vectorised_func
    
    @_vectorise
    def HubbleParameter(self, z):
        return 1./HubbleParameter(z, self._CosmologicalParameters)

    @_vectorise
    def LuminosityDistance(self, z):
        return LuminosityDistance(self._CosmologicalParameters, z)
    
    @_vectorise
    def ComovingTransverseDistance(self, z):
        return ComovingTransverseDistance(self._CosmologicalParameters, z)
     
    @_vectorise
    def ComovingLOSDistance(self, z):
        return ComovingLOSDistance(self._CosmologicalParameters, z)
     
    @_vectorise
    def UniformComovingVolumeDensity(self, z):
        return UniformComovingVolumeDensity(z, self._CosmologicalParameters)
    
    @_vectorise
    def ComovingVolumeElement(self, z):
        return ComovingVolumeElement(z, self._CosmologicalParameters)
    
    @_vectorise
    def ComovingVolume(self, z):
        return ComovingVolume(self._CosmologicalParameters, z)
    
    @_vectorise
    def dDTdDC(self, DC):
        if self.ok == 0.:
            return 1.
        elif self.ok > 0.:
            return np.cosh(np.sqrt(self.ok)*DC/self.HubbleDistance)
        else:
            return np.cos(np.sqrt(-self.ok)*DC/self.HubbleDistance)
    
    @_vectorise
    def dDLdz(self, z):
        DC   = self.ComovingLOSDistance(z)
        DT   = self.ComovingTransverseDistance(z)
        invE = 1./self.HubbleParameter(z)
        return DT + (1.+z)*self.dDTdDC(DC)*self.HubbleDistance*invE
    
    @_vectorise
    def Redshift(self, DL):
        if DL == 0.:
            return 0.
        else:
            def objective(z, self, DL):
                return DL - self.LuminosityDistance(z)
            return newton(objective,1.0,args=(self, DL))

class CosmologicalParameters_astropy:
    """
    Wrapper for Astropy functions in a single class.
    
    Arguments:
        double h:   normalised hubble constant h = H0/100 km/Mpc/s
        double om:  matter energy density
        double ol:  cosmological constant density
        double w0:  0th order dark energy equation of state parameter
        double w1:  1st order dark energy equation of state parameter (backward compatibility)
        double w2:  2nd order dark energy equation of state parameter (backward compatibility)
        
    Returns:
        CosmologicalParameters: instance of CosmologicalParameters class
    """
    def __init__(self, h, om, ol, w0, w1 = 0, w2 = 0):
        self.h = h
        self.om = om
        self.ol = ol
        self.ok = 1.-om-ol
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.Cosmology = wCDM(H0 = self.h*100, Om0 = self.om, Ode0 = self.ol, w0 = self.w0)
        self.HubbleDistance = 2.99792458e5/(100*self.h)
    
    def HubbleParameter(self, z):
        return self.Cosmology.H(z).value / (self.h*100)

    def LuminosityDistance(self, z):
        return self.Cosmology.luminosity_distance(z).value
    
    def ComovingTransverseDistance(self, z):
        return self.Cosmology.comoving_transverse_distance(z).value
     
    def ComovingLOSDistance(self, z):
        return self.Cosmology.comoving_distance(z).value
    
    def ComovingVolumeElement(self, z):
        return self.Cosmology.differential_comoving_volume(z).value
    
    def ComovingVolume(self, z):
        return self.Cosmology.comoving_volume(z).value
    
    def Redshift(self, DL):
        return z_at_value(self.Cosmology.luminosity_distance, Quantity(DL, 'Mpc')).value
    
    def dDLdz(self, z):
        DC   = self.ComovingLOSDistance(z)
        DT   = self.ComovingTransverseDistance(z)
        invE = 1./self.HubbleParameter(z)
        return DT + (1.+z)*self.dDTdDC(DC)*self.HubbleDistance*invE
    
    def dDTdDC(self, DC):
        if self.ok == 0.:
            return 1.
        elif self.ok > 0.:
            return np.cosh(np.sqrt(self.ok)*DC/self.HubbleDistance)
        else:
            return np.cos(np.sqrt(-self.ok)*DC/self.HubbleDistance)

Planck18 = CosmologicalParameters(0.674, 0.315, 0.685, -1)
Planck15 = CosmologicalParameters(0.679, 0.3065, 0.6935, -1)

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
