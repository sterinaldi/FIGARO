# encoding: utf-8
# cython: profile=False
# cython: linetrace=False
# cython: language_level=3, boundscheck=False, wraparound=False, binding=True, embedsignature=True

from __future__ import division
from libc.math cimport M_SQRT2, M_PI, erf, log, fabs
cimport cython
from cpnest.parameter cimport LivePoint
from scipy.special.cython_special cimport erfinv

def transform_to_probit(LivePoint x, list bounds):
    return _transform_to_probit(x, bounds)

cdef LivePoint _transform_to_probit(LivePoint x, list bounds):
    '''
    Coordinate change into probit space.
    cdf_normal is the cumulative distribution function of the unit normal distribution.
    WARNING: returns NAN if x is not in [xmin, xmax].
    
    t(x) = cdf^-1_normal((x-x_min)/(x_max - x_min))

    
    Arguments:
        :float or np.ndarray samples: sample(s) to transform
    Returns:
        :float or np.ndarray: sample(s)
    '''
    cdef unsigned int i
    cdef unsigned int n = x.dimension
    cdef LivePoint o = x._copy()
    cdef double cdf
    for i in range(n):
        cdf = (x.values[i] - bounds[i][0])/(bounds[i][1]-bounds[i][0])
        o.values[i] = M_SQRT2*erfinv(2*cdf-1)
    return o

def transform_from_probit(LivePoint x, list bounds):
    return _transform_from_probit(x, bounds)

cdef LivePoint _transform_from_probit(LivePoint x, list bounds):
    '''
    Coordinate change from probit to natural space.
    cdf_normal is the cumulative distribution function of the unit normal distribution.
    
    x(t) = xmin + (xmax-xmin)*cdf_normal(t|0,1)
    
    Arguments:
        :float or np.ndarray samples: sample(s) to antitransform
    Returns:
        :float or np.ndarray: sample(s)
        
        1/2[1 + erf(z/sqrt(2))].
    '''
    cdef unsigned int i
    cdef unsigned int n = x.dimension
    cdef LivePoint o = x._copy()
    cdef double cdf
    for i in range(n):
        cdf = 0.5*(1.0+erf(x.values[i]/M_SQRT2))
        o.values[i] = bounds[i][0]+(bounds[i][1]-bounds[i][0])*cdf
    return o

def probit_logJ(LivePoint x, list bounds):
    """
    Returns the log Jacobian of the probit
    trasformation
    ----------
    Parameter:
        point: :obj:`cpnest.parameter.LivePoint`
    ----------
    Returns:
        res: :obj:`double`
            The values of the log Jacobian
    """
    return _probit_logJ(x, bounds)

cdef double _probit_logJ(LivePoint x, list bounds):
    cdef unsigned int i
    cdef unsigned int n = x.dimension
    cdef double res = 0.0
    cdef double log2PI = log(2.0*M_PI)
    for i in range(n):
        res += -0.5*x.values[i]**2-0.5*log2PI-log(bounds[i][1]-bounds[i][0])
    return res

def transform_to_hypercube(LivePoint x, list bounds):
    return _transform_to_hypercube(x, bounds)

cdef LivePoint _transform_to_hypercube(LivePoint x, list bounds):
    """
    Maps the bounds of the parameters onto [0,1]
    ----------
    Parameter:
        point: :obj:`cpnest.parameter.LivePoint`
    ----------
    Returns:
        normalised_value: :obj:`array.array`
            The values of the parameter mapped into the Ndim-cube
    """
    cdef unsigned int i
    cdef unsigned int n = x.dimension
    cdef LivePoint o = x._copy()
    for i in range(n):
        o.values[i] = (x.values[i]-bounds[i][0])/(bounds[i][1]-bounds[i][0])
    return o

def transform_from_hypercube(LivePoint x, list bounds):
    return _transform_from_hypercube(x, bounds)

cdef LivePoint _transform_from_hypercube(LivePoint x, list bounds):
    """
    Maps from [0,1]^Ndim to the full range of the parameters
    Inverse of to_normalised()
    ----------
    Parameter:
        normalised_vaue: array-like values in range (0,1)
    ----------
    Returns:
        point: :obj:`cpnest.parameter.LivePoint`
    """
    cdef unsigned int i
    cdef unsigned int n = x.dimension
    cdef LivePoint o = x._copy()

    for i in range(n):
        o.values[i] = bounds[i][0]+(bounds[i][1]-bounds[i][0])*x.values[i]
    return o

def hypercube_logJ(list bounds):
    """
    Returns the log Jacobian of the hypercube
    trasformation
    ----------
    Parameter:
        bounds: :obj:`list`
    ----------
    Returns:
        res: :obj:`double`
            The values of the log Jacobian
    """
    return _hypercube_logJ(bounds)

cdef double _hypercube_logJ(list bounds):
    cdef unsigned int i
    cdef unsigned int n = len(bounds)
    cdef double res = 0.0
    for i in range(n):
        res += 1./(bounds[i][1]-bounds[i][0])
    return fabs(res)
