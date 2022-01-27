from tqdm import tqdm
from scipy.stats import multivariate_normal as mn
from corner import corner
from online_skyloc.coordinates import celestial_to_cartesian, cartesian_to_celestial

samples = np.genfromtxt('data/GW150914_full_volume.txt')

cart_samples = celestial_to_cartesian(samples))
bounds = np.array([[-1000,1000], [-1000,1000], [-1000,1000]])
n_samps = len(samples)

DPGMM = mixture(prior_pars, bounds)
for s in tqdm(cart_samples):
    DPGMM.add_new_point(np.atleast_2d(s))
    
samps_mix = DPGMM.sample_from_dpgmm(n_samps)

samps_mix = cartesian_to_celestial(samps_mix)
samps_back = cartesian_to_celestial(cart_samples)

c = corner(samps_back, color = 'orange', hist_kwargs={'density':True})
c = corner(samps_mix, fig = c, color = 'blue', hist_kwargs={'density':True})

plt.savefig('data/GW150914_full_volume.pdf', bbox_inches = 'tight')




