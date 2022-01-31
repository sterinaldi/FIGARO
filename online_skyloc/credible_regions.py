import numpy as np
from online_skyloc.cumulative import fast_log_cumulative
import healpy as hp

def FindHeights(args):
    (sortarr,  cumarr, level) = args
    return sortarr[np.abs(cumarr-np.log(level)).argmin()]

def ConfidenceVolume(log_volume_map, n_points_area, distance_grid, area_grid, adLevels = [0.68, 0.90]):
    # create a normalized cumulative distribution
    log_volume_map_sorted = np.sort(log_volume_map.flatten())[::-1]
    log_volume_map_cum = fast_log_cumulative(log_volume_map_sorted)
    
    # find the indeces  corresponding to the given CLs
    adLevels = np.ravel([adLevels])
    args = [(log_volume_map_sorted, log_volume_map_cum, level) for level in adLevels]
    adHeights = [FindHeights(a) for a in args]
    heights = {str(lev):hei for lev,hei in zip(adLevels,adHeights)}
    dd = np.diff(distance_grid)[0]
    dA = hp.nside2pixarea(n_points_area, degrees = False)
    volumes = []
    index   = []
    for height in adHeights:
        index_v = np.array(np.where(log_volume_map > height))
        volumes.append(np.sum([distance_grid[i_d]**2. * dd * dA for i_d in index_d[:,2]]))
        index.append(index_v.T)
    volume_confidence = np.array(volumes)
    
    return volume_confidence, index, adHeights

def ConfidenceArea(log_skymap, n_points_area, adLevels = [0.68, 0.90]):
    
    # create a normalized cumulative distribution
    log_skymap_sorted = np.sort(log_skymap.flatten())[::-1]
    log_skymap_cum = fast_log_cumulative(log_skymap_sorted)
    # find the indeces  corresponding to the given CLs
    adLevels = np.ravel([adLevels])
    args = [(log_skymap_sorted, log_skymap_cum, level) for level in adLevels]
    adHeights = [FindHeights(a) for a in args]
    dA = hp.nside2pixarea(n_points_area, degrees = True)
    areas = []
    index = []
    for height in adHeights:
        index_hp = np.array(np.where(log_skymap > height))
        areas.append(len(index_hp)*dA)
        index.append(index_hp.T)
    area_confidence = np.array(areas)
    
    return area_confidence, index, adHeights

def ConfidenceDistance(distance_map, distance_grid, adLevels = [0.68, 0.90]):
    dd = np.diff(distance_grid)[0]
    cumulative_distribution = np.cumsum(distance_map*dd)
    distances = []
    index     = []
    for cl in adLevels:
        idx = np.abs(cumulative_distribution-cl).argmin()
        distances.append(distance_grid_grid[idx])
        index.append(idx.T)
    distance_confidence = np.array(distances)

    return distance_confidence, index
