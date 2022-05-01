import numpy as np
from figaro.cumulative import fast_log_cumulative
from scipy.special import logsumexp

# -----------------------
# confidence calculations
# -----------------------
def FindNearest(ra, dec, dist, value):
    """
    Find the pixel that contains the triplet (ra', dec', D') stored in value.
    
    Arguments:
        :np.ndarray ra:   right ascension values used to build the grid
        :np.ndarray dec:  declination values used to build the grid
        :np.ndarray dist: luminosity distance values used to build the grid
        :iterable value:  triplet to locate (ra', dec', D')
    
    Returns:
        :np.ndarray: grid indices of pixel
    """
    idx = np.zeros(3, dtype = int)
    for i, (d, v) in enumerate(zip([ra, dec, dist], value)):
        idx[i] = int(np.abs(d-v).argmin())
    return idx

def FindHeights(args):
    """
    Find height correspinding to a certain credible level given a sorted array of probabilities and the corresponding cumulative
    
    Arguments:
        :tuple args: tuple containing the sorted array, the cumulative array and a double corresponding to the credible level
    
    Returns:
        :double: height corresponding to the credible level
    """
    (sortarr,cumarr,level) = args
    return sortarr[np.abs(cumarr-np.log(level)).argmin()]

def FindHeightForLevel(inLogArr, adLevels, logdd):
    """
    Given a probability array, computes the heights corresponding to some given credible levels.
    
    Arguments:
        :np.ndarray inLogArr: probability array
        :iterable adLevels:   credible levels
        :double logdd:        variables log differential (∑ log(dx_i))
        
    Returns:
        :np.ndarray: heights corresponding to adLevels
    """
    # flatten and create reversed sorted list
    adSorted = np.ascontiguousarray(np.sort(inLogArr.flatten())[::-1])
    # create a normalized cumulative distribution
    adCum = fast_log_cumulative(adSorted + logdd)
    # find values closest to levels
    adHeights = []
    adLevels = np.ravel([adLevels])
    for level in adLevels:
        idx = (np.abs(adCum-np.log(level))).argmin()
        adHeights.append(adSorted[idx])
    adHeights = np.array(adHeights)
    return adHeights

def FindLevelForHeight(inLogArr, logvalue, logdd):
    """
    Given a probability array, computes the credible levels corresponding to a given height.
    
    Arguments:
        :np.ndarray inLogArr: log probability array
        :double logvalue:     height
        :double logdd:        variables log differential (∑ log(dx_i))
    
    Returns:
        :np.ndarray: credible level corresponding to logvalue
    """
    # flatten and create reversed sorted list
    adSorted = np.ascontiguousarray(np.sort(inLogArr.flatten())[::-1])
    # create a normalized cumulative distribution
    adCum = fast_log_cumulative(adSorted + logdd)
    # find index closest to value
    idx = (np.abs(adSorted-logvalue)).argmin()
    return np.exp(adCum[idx])

def ConfidenceVolume(log_volume_map, log_measure, ra_grid, dec_grid, distance_grid, adLevels = [0.50, 0.90]):
    """
    Compute the credible volume(s) for a 3D probability distribution
    
    Arguments:
        :np.ndarray log_volume_map: probability density for each pixel
        :np.ndarray ra_grid:        right ascension values used to build the grid
        :np.ndarray dec_grid:       declination values used to build the grid
        :np.ndarray distance_grid:  luminosity distance values used to build the grid
        :iterable adLevels:         credible level(s)
    
    Returns:
        :np.ndarray: credible volume(s)
        :iterable:   indices of pixels within credible volume(s)
        :np.ndarray: height(s) corresponding to credible volume(s)
    """
    dd  = np.diff(distance_grid)[0]
    ddec = np.diff(dec_grid)[0]
    dra = np.diff(ra_grid)[0]
    # create a normalized cumulative distribution
    log_volume_map_sorted = np.ascontiguousarray(np.sort(log_volume_map.flatten())[::-1])
    log_measure_sorted    = np.ascontiguousarray(log_measure.flatten()[np.argsort(log_volume_map.flatten())][::-1])
    log_volume_map_cum    = fast_log_cumulative(log_volume_map_sorted + log_measure_sorted + np.log(dra) + np.log(ddec) + np.log(dd))
    # find the indeces  corresponding to the given CLs
    adLevels = np.ravel([adLevels])
    args = [(log_volume_map_sorted, log_volume_map_cum, level) for level in adLevels]
    adHeights = [FindHeights(a) for a in args]
    volumes         = []
    index           = []
    for height in adHeights:
        (i_ra, i_dec, i_d,) = np.where(log_volume_map>=height)
        volumes.append(np.sum([distance_grid[i_d]**2. *np.cos(dec_grid[i_dec]) * dd * dra * ddec for i_d,i_dec in zip(i_d,i_dec)]))
        index.append(np.array([i_ra, i_dec, i_d]).T)
    volume_confidence = np.array(volumes)
    
    return volume_confidence, index, np.array(adHeights)

def ConfidenceArea(log_skymap, log_measure, ra_grid, dec_grid, adLevels = [0.50, 0.90]):
    """
    Compute the credible area(s) for a 2D probability distribution
    
    Arguments:
        :np.ndarray log_skymap: probability density for each pixel
        :np.ndarray ra_grid:    right ascension values used to build the grid
        :np.ndarray dec_grid:   declination values used to build the grid
        :iterable adLevels:     credible level(s)
    
    Returns:
        :np.ndarray: credible area(s)
        :iterable:   indices of pixels within credible area(s)
        :np.ndarray: height(s) corresponding to credible area(s)
    """
    # create a normalized cumulative distribution
    ddec = np.diff(dec_grid)[0]
    dra = np.diff(ra_grid)[0]
    log_skymap_sorted = np.ascontiguousarray(np.sort(log_skymap.flatten())[::-1])
    log_measure_sorted = np.ascontiguousarray(log_measure.flatten()[np.argsort(log_skymap.flatten())][::-1])
    log_skymap_cum = fast_log_cumulative(log_skymap_sorted + log_measure_sorted.flatten() + np.log(dra) + np.log(ddec))
    # find the indeces  corresponding to the given CLs
    adLevels = np.ravel([adLevels])
    args = [(log_skymap_sorted, log_skymap_cum, level) for level in adLevels]
    adHeights = [FindHeights(a) for a in args]
    areas = []
    index = []
    for height in adHeights:
        (i_ra,i_dec,) = np.where(log_skymap>=height)
        areas.append(np.sum([dra*np.cos(dec_grid[i_d])*ddec for i_d in i_dec])*(180.0/np.pi)**2.0)
        index.append(np.array([i_ra, i_dec]).T)
    area_confidence = np.array(areas)
    
    return area_confidence, index, np.array(adHeights)

def ConfidenceInterval(probability, measure, grid, adLevels = [0.50, 0.90]):
    """
    Compute the credible interval(s) for a 1D probability distribution
    
    Arguments:
        :np.ndarray probability: probability density for each pixel
        :np.ndarray grid:        values used to build the grid
        :iterable adLevels:      credible level(s)
    
    Returns:
        :np.ndarray: credible interval(s)
        :iterable:   indices of bins within credible area(s)
    """
    dx = np.diff(grid)[0]
    cumulative_distribution = np.cumsum(probability*dx*measure)
    values = []
    index  = []
    for cl in adLevels:
        idx = np.abs(cumulative_distribution-cl).argmin()
        values.append(grid[idx])
        index.append(idx)
    values_confidence = np.array(values)

    return values_confidence, index

