import unittest
import numpy as np
from figaro.mixture import DPGMM, HDPGMM, mixture
from figaro.utils import get_priors
from figaro.diagnostic import compute_entropy_single_draw

class TestFIGARO(unittest.TestCase):
    """
    Class to test (H)/DPGMM behaviour (Gaussian distribution, known entropy)
    """
    def test_DPGMM(self):
        generator = np.random.default_rng(seed = 42)
        mean, std = (0., 1.)
        bounds    = np.atleast_2d([-5,5])
        # Generate samples
        samples   = generator.normal(mean, std, size = 3000)
        # Reconstruction
        priors    = get_priors(bounds, samples = samples, hierarchical = False, probit = False)
        mix       = DPGMM(bounds, prior_pars = priors, probit = False)
        d         = mix.density_from_samples(samples)
        # Check entropy
        S, dS     = compute_entropy_single_draw(d, return_error = True)
        exp_S     = np.log2(std*np.sqrt(2*np.pi*np.e))
        self.assertTrue(np.abs(S-exp_S)/dS < 3.)
    
    def test_HDPGMM(self):
        generator   = np.random.default_rng(seed = 42)
        mean, std   = (0., 1.)
        std_obs     = 0.1
        bounds      = np.atleast_2d([-5,5])
        # Generate mock observations
        samples     = generator.normal(mean, std, size = 3000)
        obs_s       = np.array([generator.normal(s, std_obs, size = 1) for s in samples])
        all_samples = np.array([generator.normal(s, std_obs, size = 1000) for s in obs_s])
        se_draws    = np.array([[mixture(means     = np.atleast_2d([s]),
                                         covs      = np.atleast_3d([std_obs**2]),
                                         w         = np.atleast_1d([1.]),
                                         probit    = False,
                                         bounds    = bounds,
                                         dim       = 1,
                                         n_cl      = 1,
                                         n_pts     = 1000,
                                         make_comp = False)] for s in obs_s])
        # Reconstruction
        priors      = get_priors(bounds, samples = all_samples, hierarchical = True, probit = False)
        mix         = HDPGMM(bounds, prior_pars = priors, probit = False)
        d           = mix.density_from_samples(se_draws)
        # Check entropy
        S, dS       = compute_entropy_single_draw(d, return_error = True)
        exp_S       = np.log2(std*np.sqrt(2*np.pi*np.e))
        self.assertTrue(np.abs(S-exp_S)/dS < 3.)
    
if __name__ == '__main__':
    unittest.main()
