import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from corner import corner
from online_skyloc.threeDvolume import VolumeReconstruction
from online_skyloc.coordinates import celestial_to_cartesian, cartesian_to_celestial

samples = np.genfromtxt('data/GW150914_full_volume.txt')

np.random.shuffle(samples)

DPGMM = VolumeReconstruction(max_dist = 1000, name = 'GW150914', glade_file = '/Users/stefanorinaldi/Documents/p_z/glade+.hdf5',  incr_plot = False)#, n_gridpoints = [100,100,100])
DPGMM.density_from_samples(samples)
