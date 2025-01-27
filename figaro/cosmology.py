import numpy as np
from numba import njit
from scipy.interpolate import interp1d
from astropy.cosmology import wCDM, z_at_value
from astropy.units import Quantity

class CosmologicalParameters:
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

    def HubbleParameter(self, z):
        return self.Cosmology.H(z).value

    def LuminosityDistance(self, z):
        return self.cosmology.luminosity_distance(z).value
    
    def ComovingTransverseDistance(self, z):
        return self.Cosmology.comoving_transverse_distance(z).value
     
    def ComovingLOSDistance(self, z):
        return self.Cosmology.comoving_distance(z).value
    
    def ComovingVolumeElement(self, z):
        return self.Cosmology.differential_comoving_volume(z).value
    
    def ComovingVolume(self, z):
        return self.Cosmology.comoving_volume(z).value
    
    def Redshift(self, DL):
        if DL == 0.:
            return 0.
        else:
            return z_at_value(self.Cosmology.luminosity_distance, Quantity(DL, 'Mpc'))

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
