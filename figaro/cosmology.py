import numpy as np
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio") # Silence LAL warnings with ipython
from scipy.optimize import newton
from lal._lal import LuminosityDistance, UniformComovingVolumeDensity, ComovingVolumeElement, ComovingVolume, CreateCosmologicalParameters

class CosmologicalParameters:
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
    def __init__(self, h, om, ol, w0, w1, w2):
        self.h = h
        self.om = om
        self.ol = ol
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self._CosmologicalParameters = CreateCosmologicalParameters(self.h, self.om, self.ol, self.w0, self.w1, self.w2)
    
    def _vectorise(func):
        def vectorised_func(self, x):
            if hasattr(x, "__iter__"):
                return np.array([func(self, xi) for xi in x])
            else:
                return func(self, x)
        return vectorised_func
    
    @_vectorise
    def LuminosityDistance(self, z):
        return LuminosityDistance(self._CosmologicalParameters, z)
     
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
    def Redshift(self, DL):
        if DL == 0.:
            return 0.
        else:
            def objective(z, self, DL):
                return DL - self.LuminosityDistance(z)
            return newton(objective,1.0,args=(self, DL))

Planck18 = CosmologicalParameters(0.674, 0.315, 0.685, -1, 0, 0)
