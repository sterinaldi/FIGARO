import numpy as np
from figaro.mixture import mixture
from figaro.exceptions import FIGAROException

def _marginalise(mix, axis = -1):
    ax     = np.atleast_1d(axis)
    means  = np.delete(mix.means, ax, axis = -1)
    covs   = np.delete(np.delete(mix.covs, ax, axis = -1), ax, axis = -2)
    dim    = mix.dim - len(ax)
    if dim < 1:
        raise FIGAROException("Cannot marginalise out all dimensions")
    bounds = np.delete(mix.bounds, ax, axis = 0)
    
    return mixture(means, covs, mix.w, bounds, dim, mix.n_cl, mix.n_pts)

def marginalise(draws, axis = -1):
    if iterable(draws):
        return np.array([_marginalise(d, axis) for d in draws])
    else:
        return _marginalise(draws, axis)
