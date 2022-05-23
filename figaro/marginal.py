import numpy as np
from figaro.mixture import mixture
from figaro.exceptions import FIGAROException

def _marginalise(mix, axis = -1):
    """
    Marginalises out one or more dimensions from a FIGARO draw.
    
    Arguments:
        :figaro.mixture.mixture draws: mixture
        :int or list of int axis:      axis to marginalise on
    
    Returns:
        :figaro.mixture.mixture: the marginalised mixture
    """
    ax     = np.atleast_1d(axis)
    dim    = mix.dim - len(ax)
    if dim < 1:
        raise FIGAROException("Cannot marginalise out all dimensions")
    means  = np.delete(mix.means, ax, axis = -1)
    covs   = np.delete(np.delete(mix.covs, ax, axis = -1), ax, axis = -2)
    bounds = np.delete(mix.bounds, ax, axis = 0)
    
    return mixture(means, covs, mix.w, bounds, dim, mix.n_cl, mix.n_pts)

def marginalise(draws, axis = -1):
    """
    Marginalises out one or more dimensions from a FIGARO draw.
    
    Arguments:
        :figaro.mixture.mixture draws: mixture(s)
        :int or list of int axis:      axis to marginalise on
    
    Returns:
        :figaro.mixture.mixture: the marginalised mixture(s)
    """
    if axis == []:
        return draws
    if np.iterable(draws):
        return np.array([_marginalise(d, axis) for d in draws])
    else:
        return _marginalise(draws, axis)
