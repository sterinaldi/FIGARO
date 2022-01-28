import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from corner import corner
from online_skyloc.incremental_gibbs import VolumeReconstruction
from online_skyloc.coordinates import celestial_to_cartesian, cartesian_to_celestial

samples = np.genfromtxt('data/GW150914_full_volume.txt')

DPGMM = VolumeReconstruction(max_dist = 1000, name = 'GW150914')
DPGMM.density_from_samples(samples)



