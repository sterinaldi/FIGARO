import numpy as np
cimport numpy as np
cdef extern from "lal/LALCosmologyCalculator.h" nogil:
    # cosmological parameters structure
    ctypedef struct LALCosmologicalParameters:
        double h;
        double om;
        double ol;
        double ok;
        double w0;
        double w1;
        double w2;

    cdef double XLALLuminosityDistance(
            LALCosmologicalParameters *omega,
            double z)

    cdef double XLALAngularDistance(
            LALCosmologicalParameters *omega,
            double z)

    cdef double XLALComovingLOSDistance(
            LALCosmologicalParameters *omega,
            double z)

    cdef double XLALComovingTransverseDistance(
            LALCosmologicalParameters *omega,
            double z)

    cdef double XLALHubbleDistance(
            LALCosmologicalParameters *omega
            )

    cdef double XLALHubbleParameter(double z,
            void *omega
            )

    cdef double XLALIntegrateHubbleParameter(LALCosmologicalParameters *omega, double z)

    cdef double XLALUniformComovingVolumeDistribution(LALCosmologicalParameters *omega, double z, double zmax)

    cdef double XLALUniformComovingVolumeDensity(double z, void *omega)

    cdef double XLALIntegrateComovingVolumeDensity(LALCosmologicalParameters *omega, double z)

    cdef double XLALIntegrateComovingVolume(LALCosmologicalParameters *omega, double z)

    cdef double XLALComovingVolumeElement(double z, void *omega)

    cdef double XLALComovingVolume(LALCosmologicalParameters *omega, double z)

    cdef LALCosmologicalParameters *XLALCreateCosmologicalParameters(double h, double om, double ol, double w0, double w1, double w2)

    cdef void XLALDestroyCosmologicalParameters(LALCosmologicalParameters *omega)

    cdef double XLALGetHubbleConstant(LALCosmologicalParameters *omega)

    cdef double XLALGetOmegaMatter(LALCosmologicalParameters *omega)

    cdef double XLALGetOmegaLambda(LALCosmologicalParameters *omega)

    cdef double XLALGetOmegaK(LALCosmologicalParameters *omega)

    cdef double XLALGetW0(LALCosmologicalParameters *omega)

    cdef double XLALGetW1(LALCosmologicalParameters *omega)

    cdef double XLALGetW2(LALCosmologicalParameters *omega)
    
cdef class CosmologicalParameters:
    cdef LALCosmologicalParameters* _LALCosmologicalParameters
    cdef public double h
    cdef public double om
    cdef public double ol
    cdef public double w0
    cdef public double w1
    cdef public double _HubbleParameter(self,double z) nogil
    cdef public double _LuminosityDistance_double(self, double z) nogil
    cdef public np.ndarray[double, ndim=1, mode="c"] _LuminosityDistance(self, np.ndarray[double, ndim=1, mode="c"] z)
    cdef public double _HubbleDistance(self) nogil
    cdef public double _IntegrateComovingVolumeDensity(self, double zmax) nogil
    cdef public double _IntegrateComovingVolume(self, double zmax) nogil
    cdef public double _UniformComovingVolumeDensity_double(self, double z) nogil
    cdef public np.ndarray[double, ndim=1, mode="c"] _UniformComovingVolumeDensity(self, np.ndarray[double, ndim=1, mode="c"] z)
    cdef public double _UniformComovingVolumeDistribution(self, double z, double zmax) nogil
    cdef public double _ComovingVolumeElement_double(self,double z) nogil
    cdef public np.ndarray[double, ndim=1, mode="c"] _ComovingVolumeElement(self, np.ndarray[double, ndim=1, mode="c"] z)
    cdef public double _ComovingVolume_double(self, double z) nogil
    cdef public np.ndarray[double, ndim=1, mode="c"] _ComovingVolume(self, np.ndarray[double, ndim=1, mode="c"] z)
    cdef void _DestroyCosmologicalParameters(self) nogil
