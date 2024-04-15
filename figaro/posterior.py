import numpy as np
from scipy.stats import dirichlet, multivariate_normal as mn
from figaro.mixture import mixture, HDPGMM
from figaro.transform import *
from figaro._numba_functions import *

class sampler(HDPGMM):
    """"
    Class to reconstruct the posterior distribution given a set of samples from a prior distribution.
    Child of HDPGMM class.
    
    Arguments:
        iterable bounds:             boundaries of the rectangle over which the distribution is defined. It should be in the format [[xmin, xmax],[ymin, ymax],...]
        iterable prior_pars:         IW parameters
        double alpha0:               initial guess for concentration parameter
        double MC_draws:             number of MC draws for integral
        bool probit:                 whether to use the probit transformation or not
        callable log_likelihood: selection function approximant or samples
    
    Returns:
        HDPGMM: instance of HDPGMM class
    """
    def __init__(self, bounds,
                       log_likelihood,
                       prior_samples = None,
                       n_samples     = None,
                       alpha0        = 1.,
                       prior_pars    = None,
                       MC_draws      = None,
                       probit        = True,
                       ):
        self.log_likelihood = log_likelihood
        if prior_samples is not None:
            self.draw_samples = False
        else:
            self.draw_samples = True
            if n_samples is None:
                self.n_samples = int(1e3*len(np.atleast_2d(bounds)))
            else:
                self.n_samples = int(n_samples)
        self.logL_max = 0.
        super().__init__(bounds     = bounds,
                         alpha0     = alpha0,
                         probit     = probit,
                         prior_pars = prior_pars,
                         MC_draws   = MC_draws,
                         )
        self._draw_prior_samples(samples = prior_samples)
        self.initialise()

    def _draw_prior_samples(self, samples = None):
        """
        Draws a new set of prior samples and instantiates the corresponding mixtures
        
        Arguments:
            np.ndarray samples: prior samples (optional)
        """
        if samples is None:
            self.samples = np.random.uniform(*self.bounds.T, size = (self.n_samples, self.dim))
        else:
            self.samples = samples
        if self.probit:
            self.samples = transform_to_probit(self.samples, self.bounds)
        self.mixtures = [[mixture(np.atleast_2d(s), np.array([np.identity(self.dim)*0.]), np.ones(1), self.bounds, self.dim, 1, 0, probit = self.probit, alpha = 1., make_comp = False)] for s in self.samples]
        self.logL_max = np.max(-self.log_likelihood(self.samples))
    
    def _draw_MC_samples(self):
        """
        Draws MC samples for mu and sigma and computes log alpha factor
        """
        super()._draw_MC_samples()
        if self.probit:
            self.log_alpha_factor = np.array([logsumexp_jit(-self.log_likelihood(transform_from_probit(mn(m,s).rvs(self.MC_draws), self.bounds))) - np.log(self.MC_draws) for m, s in zip(self.mu_MC, self.sigma_MC)])
        else:
            self.log_alpha_factor = np.array([logsumexp_jit(-self.log_likelihood(mn(m,s).rvs(self.MC_draws))) - np.log(self.MC_draws) for m, s in zip(self.mu_MC, self.sigma_MC)])
        self.log_alpha_factor = np.nan_to_num(self.log_alpha_factor, nan = np.inf, posinf = np.inf, neginf = np.inf)
    
    def density_from_samples(self, make_comp = True):
        """
        Reconstruct the posterior probablity density from a set of prior samples.
        
        Arguments:
            bool make_comp:   whether to instantiate the scipy.stats.multivariate_normal components or not
        
        Returns:
            mixture: the inferred mixture
        """
        return super().density_from_samples(self.mixtures, make_comp = make_comp)
    
    def initialise(self):
        """
        Initialise the mixture to initial conditions
        """
        if self.draw_samples:
            self._draw_prior_samples()
        super().initialise()

    def build_mixture(self, make_comp = True):
        """
        Instances a mixture class representing the inferred distribution
        
        Arguments:
            bool make_comp:   whether to instantiate the scipy.stats.multivariate_normal components or not
        
        Returns:
            mixture: the inferred distribution
        """
        if self.n_cl == 0:
            raise FIGAROException("You are trying to build an empty mixture - perhaps you called the initialise() method. If you are using the density_from_samples() method, the inferred mixture is returned by that method as an instance of mixture class.")
        idx  = np.where(np.array(self.N_list) > 0)[0]
        logN = np.array([comp.log_N_true for comp in np.array(self.mixture)[idx]])
        Npts = np.exp(logN - logsumexp_jit(logN)) * self.n_pts/np.mean(np.exp(self.log_likelihood(self.rvs(self.MC_draws))))
        w   = dirichlet(Npts + self.alpha/self.n_cl).rvs()[0]
        return mixture(np.array([comp.mu for comp in np.array(self.mixture)[idx]]), np.array([comp.sigma for comp in np.array(self.mixture)[idx]]), w, self.bounds, self.dim, (np.array(self.N_list) > 0).sum(), self.n_pts, self.alpha, probit = self.probit, make_comp = make_comp)
