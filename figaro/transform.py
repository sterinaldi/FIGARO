import numpy as np
from scipy.special import erfinv, erf

log2PI = np.log(2.0*np.pi)

"""
Scipy's implementation of ERF: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erf.html
See notes there for cumulative of the unit normal distribution.
"""

def transform_to_probit(x, bounds):
    """
    Coordinate change into probit space.
    cdf_normal is the cumulative distribution function of the unit normal distribution.
    WARNING: returns NAN if x is not in [xmin, xmax].
    
    t(x) = cdf^-1_normal((x-x_min)/(x_max - x_min))

    
    Arguments:
        np.ndarray x:      sample(s) to transform (2d array)
        np.ndarray bounds: limits for each dimension (2d array, [[xmin, xmax], [ymin, ymax]...])
        
    Returns:
        np.ndarray: sample(s)
    """
    dbounds = bounds[:,1]-bounds[:,0]
    sigma   = dbounds*0.34
    cdf = (x - bounds[:,0])/dbounds
    o = np.sqrt(2.0)*erfinv(2*cdf-1)*sigma
    return o

def transform_from_probit(x, bounds):
    """
    Coordinate change from probit to natural space.
    cdf_normal is the cumulative distribution function of the unit normal distribution.
    
    x(t) = xmin + (xmax-xmin)*cdf_normal(t|0,1)
    
    Arguments:
        np.ndarray x:      sample(s) to antitransform (2d array)
        np.ndarray bounds: limits for each dimension (2d array, [[xmin, xmax], [ymin, ymax]...])
        
    Returns:
        np.ndarray: sample(s)
    """
    dbounds = bounds[:,1]-bounds[:,0]
    sigma   = dbounds*0.34
    cdf = 0.5*(1.0+erf(x/(np.sqrt(2.0)*sigma)))
    o = bounds[:,0]+dbounds*cdf
    return o

def probit_log_jacobian(x, bounds, flag = True):
    """
    Jacobian of the probit transformation
    
    Arguments:
        np.ndarray x:      sample(s) to evaluate the jacobian at
        np.ndarray bounds: limits for each dimension (2d array, [[xmin, xmax], [ymin, ymax]...])
        bool flag:         whether to skip the evaluation
    
    Returns:
        np.ndarray: log jacobian (zeros if flag is False)
    """
    if not flag:
        return np.zeros(len(x))
    dbounds = bounds[:,1]-bounds[:,0]
    sigma   = dbounds*0.34
    res     = -0.5*(x/sigma)**2-0.5*(log2PI)+np.log(dbounds)-np.log(sigma)
    return res

def probit_logJ(x, bounds, flag = True):
    """
    Jacobian of the probit transformation marginalised over dimensions
    
    Arguments:
        np.ndarray x:      sample(s) to evaluate the jacobian at
        np.ndarray bounds: limits for each dimension (2d array, [[xmin, xmax], [ymin, ymax]...])
        bool flag:         whether to skip the evaluation
    
    Returns:
        np.ndarray: log jacobian (zeros if flag is False)
    """
    if not flag:
        return np.zeros(len(x))
    dbounds = bounds[:,1]-bounds[:,0]
    sigma   = dbounds*0.34
    res     = np.sum(-0.5*(x/sigma)**2-0.5*log2PI+np.log(dbounds)-np.log(sigma), axis = -1)
    return res

def gradient_inv_jacobian(x, bounds, flag = True):
    """
    logarithmic gradient of the probit transformation Jacobian
    
    Arguments:
        np.ndarray x:      sample(s) to evaluate the jacobian at
        np.ndarray bounds: limits for each dimension (2d array, [[xmin, xmax], [ymin, ymax]...])
        bool flag:         whether to skip the evaluation
    
    Returns:
        np.ndarray: log jacobian (ones if flag is False)
    """
    if not flag:
        return np.ones(len(x))
    dbounds = bounds[:,1]-bounds[:,0]
    sigma   = dbounds*0.34
    return x/sigma**2.
