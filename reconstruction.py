import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from corner import corner
from online_skyloc.incremental_gibbs import mixture

data_file = 'path/to/file'
file_name = 'namefile.pdf'
samples = np.genfromtxt(data_file)
names = ['m1','m2']

bounds = np.array([[10,40],[1,5]])#,[0,1],[0,1]])
n_samps = len(samples)

DPGMM = mixture(bounds)
for s in tqdm(samples):
    DPGMM.add_new_point(np.atleast_2d(s))
    
samps_mix = DPGMM.sample_from_dpgmm(n_samps)

c = corner(samples, color = 'orange', labels = names, hist_kwargs={'density':True})
c = corner(samps_mix, fig = c, color = 'blue', labels = names, hist_kwargs={'density':True})

plt.savefig('data/'+file_name, bbox_inches = 'tight')

