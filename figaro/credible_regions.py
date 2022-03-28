import numpy as np
from figaro.cumulative import fast_log_cumulative
from scipy.special import logsumexp

# -----------------------
# confidence calculations
# -----------------------
def FindNearest(ra, dec, dist, value):
    idx = np.zeros(3, dtype = int)
    for i, (d, v) in enumerate(zip([ra, dec, dist], value)):
        idx[i] = int(np.abs(d-v).argmin())
    return idx

def FindHeights(args):
    (sortarr,cumarr,level) = args
    return sortarr[np.abs(cumarr-np.log(level)).argmin()]

def FindHeightForLevel(inLogArr, adLevels):
    # flatten and create reversed sorted list
    adSorted = np.ascontiguousarray(np.sort(inLogArr.flatten())[::-1])
    # create a normalized cumulative distribution
    adCum = fast_log_cumulative(adSorted)
    # find values closest to levels
    adHeights = []
    adLevels = np.ravel([adLevels])
    for level in adLevels:
        idx = (np.abs(adCum-np.log(level))).argmin()
        adHeights.append(adSorted[idx])
    adHeights = np.array(adHeights)
    return adHeights

def FindLevelForHeight(inLogArr, logvalue):
    # flatten and create reversed sorted list
    adSorted = np.ascontiguousarray(np.sort(inLogArr.flatten())[::-1])
    # create a normalized cumulative distribution
    adCum = fast_log_cumulative(adSorted)
    # find index closest to value
    idx = (np.abs(adSorted-logvalue)).argmin()
    return np.exp(adCum[idx])

def ConfidenceVolume(log_volume_map, ra_grid, dec_grid, distance_grid, adLevels = [0.68, 0.90]):
    # create a normalized cumulative distribution
    log_volume_map_sorted = np.ascontiguousarray(np.sort(log_volume_map.flatten())[::-1])
    log_volume_map_cum = fast_log_cumulative(log_volume_map_sorted)
    
    # find the indeces  corresponding to the given CLs
    adLevels = np.ravel([adLevels])
    args = [(log_volume_map_sorted, log_volume_map_cum, level) for level in adLevels]
    adHeights = [FindHeights(a) for a in args]
    heights = {str(lev):hei for lev,hei in zip(adLevels,adHeights)}
    dd  = np.diff(distance_grid)[0]
    ddec = np.diff(dec_grid)[0]
    dra = np.diff(ra_grid)[0]
    volumes         = []
    index           = []
    for height in adHeights:
        
        (i_ra, i_dec, i_d,) = np.where(log_volume_map>=height)
        volumes.append(np.sum([distance_grid[i_d]**2. *np.cos(dec_grid[i_dec]) * dd * dra * ddec for i_d,i_dec in zip(i_d,i_dec)]))
        index.append(np.array([i_ra, i_dec, i_d]).T)

    volume_confidence = np.array(volumes)
    
    return volume_confidence, index, np.array(adHeights)

def ConfidenceArea(log_skymap, ra_grid, dec_grid, adLevels = [0.68, 0.90]):
    
    # create a normalized cumulative distribution
    log_skymap_sorted = np.ascontiguousarray(np.sort(log_skymap.flatten())[::-1])
    log_skymap_cum = fast_log_cumulative(log_skymap_sorted)
    # find the indeces  corresponding to the given CLs
    adLevels = np.ravel([adLevels])
    args = [(log_skymap_sorted, log_skymap_cum, level) for level in adLevels]
    adHeights = [FindHeights(a) for a in args]
    ddec = np.diff(dec_grid)[0]
    dra = np.diff(ra_grid)[0]
    areas = []
    index = []
                
    for height in adHeights:
        (i_ra,i_dec,) = np.where(log_skymap>=height)
        areas.append(np.sum([dra*np.cos(dec_grid[i_d])*ddec for i_d in i_dec])*(180.0/np.pi)**2.0)

        index.append(np.array([i_ra, i_dec]).T)
    area_confidence = np.array(areas)
    
    return area_confidence, index, np.array(adHeights)

def ConfidenceInterval(probability, grid, adLevels = [0.68, 0.90]):
    dx = np.diff(grid)[0]
    cumulative_distribution = np.cumsum(probability*dx)
    values = []
    index  = []
    for cl in adLevels:
        idx = np.abs(cumulative_distribution-cl).argmin()
        values.append(grid[idx])
        index.append(idx)
    values_confidence = np.array(values)

    return values_confidence, index

