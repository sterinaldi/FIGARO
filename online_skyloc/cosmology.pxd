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

    cdef double XLALUniformComovingVolumeDistribution(
            LALCosmologicalParameters *omega,
            double z,
            double zmax)

    cdef double XLALUniformComovingVolumeDensity(
            double z,
            void *omega)

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
    cdef public double _UniformComovingVolumeDensity(self, double z) nogil
    cdef public double _UniformComovingVolumeDistribution(self, double z, double zmax) nogil
    cdef public double _ComovingVolumeElement(self,double z) nogil
    cdef public double _ComovingVolume(self,double z) nogil
    cdef void _DestroyCosmologicalParameters(self) nogil

cdef class CosmologicalRateParameters:
    cdef public double r0
    cdef public double W
    cdef public double Q
    cdef public double R
    cpdef double StarFormationDensity(self, double z)

cdef double _StarFormationDensity(const double z, const double r0, const double W, const double R, const double Q) nogil

#cpdef double RateWeightedComovingVolumeDistribution(double z, double zmin, double zmax, CosmologicalParameters omega, CosmologicalRateParameters rate, double normalisation)

#cpdef double IntegrateRateWeightedComovingVolumeDensity(double zmin, double zmax, CosmologicalParameters omega, CosmologicalRateParameters rate)

#cpdef double RateWeightedUniformComovingVolumeDensity(double z, CosmologicalParameters omega, CosmologicalRateParameters rate)

cdef double _IntegrateRateWeightedComovingVolumeDensity(const double r0, const double W, const double Q, const double R, CosmologicalParameters omega, const double zmin, const double zmax) nogil
