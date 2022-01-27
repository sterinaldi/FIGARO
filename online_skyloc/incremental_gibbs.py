import numpy as np
from scipy.special import gammaln
import matplotlib.pyplot as plt
from online_skyloc.decorators import *

class component:
    def __init__(self, x, prior):
        self.N     = 1
        self.mean  = np.atleast_2d(x)
        self.cov   = np.identity(x.shape[-1])*0.
        self.mu    = np.atleast_2d((prior.mu*prior.k + self.N*self.mean)/(prior.k + self.N))[0]
        self.sigma = np.identity(x.shape[-1])*prior.L
        self.w     = 0.
        self.normalised = False

class prior:
    def __init__(self, k, L, nu, mu):
        self.k = k
        self.L = L
        self.mu = mu
        self.nu = nu

def student_t(df, t, mu, sigma, dim, s2max = np.inf):
    """
    http://gregorygundersen.com/blog/2020/01/20/multivariate-t/
    s2max can be removed if useless
    """
    vals, vecs = np.linalg.eigh(sigma)
    vals       = np.minimum(vals, s2max)
    logdet     = np.log(vals).sum()
    valsinv    = np.array([1./v for v in vals])
    U          = vecs * np.sqrt(valsinv)
    dev        = t - mu
    maha       = np.square(np.dot(dev, U)).sum(axis=-1)

    x = 0.5 * (df + dim)
    A = gammaln(x)
    B = gammaln(0.5 * df)
    C = dim/2. * np.log(df * np.pi)
    D = 0.5 * logdet
    E = -x * np.log1p((1./df) * maha)

    return float(A - B - C - D + E)

class mixture:
    def __init__(self, prior_pars, bounds, alpha0 = 1):
        self.bounds   = bounds
        self.prior    = prior(*prior_pars)
        self.alpha    = alpha0
        self.clusters = []
        self.n_cl     = 0
        self.n_pts    = 0
        self.s2max    = (1./3.)**2

    def add_datapoint_to_component(self, x, ss):
        x = np.atleast_2d(x)
        old_mean = ss.mean
        ss.mean  = np.atleast_2d((ss.mean*(ss.N)+x)/(ss.N+1))
        ss.cov   = (ss.N*(ss.cov + np.matmul(old_mean.T, old_mean)) + np.matmul(x.T, x))/(ss.N+1) - np.matmul(ss.mean.T, ss.mean)
        ss.N     = ss.N+1
        ss.mu    = ((self.prior.mu*self.prior.k + ss.N*ss.mean)/(self.prior.k + ss.N))[0]
        ss.sigma = (np.atleast_2d(self.prior.L*self.prior.k) + ss.cov*ss.N + self.prior.k*ss.N*np.matmul((ss.mean - self.prior.mu).T, (ss.mean - self.prior.mu))/(self.prior.k + ss.N))/(self.prior.nu + ss.N)
        return ss

    def update_alpha(self, burnin = 200):
        a_old = self.alpha
        n     = self.n_pts
        K     = self.n_cl
        for _ in range(burnin+np.random.randint(100)):
            a_new = a_old + np.random.uniform(-1,1)*0.5
            if a_new > 0:
                logP_old = gammaln(a_old) - gammaln(a_old + n) + K * np.log(a_old) - 1./a_old
                logP_new = gammaln(a_new) - gammaln(a_new + n) + K * np.log(a_new) - 1./a_new
                if logP_new - logP_old > np.log(np.random.uniform()):
                    a_old = a_new
        self.alpha = a_old
        return

    def log_predictive_likelihood(self, x, ss):
        dim = x.shape[-1]
        if ss == "new":
            ss = component(np.zeros(dim), prior = self.prior)
            ss.N = 0
        mean = ss.mean
        S    = ss.cov
        N    = ss.N
        # Update hyperparameters
        k_n  = self.prior.k + N
        mu_n = np.atleast_2d((self.prior.mu*self.prior.k + N*mean)/k_n)
        nu_n = self.prior.nu + N
        L_n  = self.prior.L*self.prior.k + S*N + self.prior.k*N*np.matmul((mean - self.prior.mu).T, (mean - self.prior.mu))/k_n
        if not np.isfinite(np.log(np.linalg.eigh(L_n)[0])).all():
            print((mean, L_n))
        # Update t-parameters
        t_df    = nu_n - dim + 1
        t_shape = L_n*(k_n+1)/(k_n*t_df)
        # Compute logLikelihood
        logL = student_t(df = t_df, t = np.atleast_2d(x), mu = mu_n, sigma = t_shape, dim = dim, s2max = self.s2max)
        return logL

    def cluster_assignment_distribution(self, x):
        scores = {}
        for i in list(np.arange(self.n_cl)) + ["new"]:
            if i == "new":
                ss = "new"
            else:
                ss = self.clusters[i]
            scores[i] = self.log_predictive_likelihood(x, ss)
            if ss is "new":
                scores[i] += np.log(self.alpha)
            else:
                scores[i] += np.log(ss.N)
        scores = {cid: np.exp(score) for cid, score in scores.items()}
        normalization = 1/sum(scores.values())
        scores = {cid: score*normalization for cid, score in scores.items()}
        return scores

    def assign_to_cluster(self, x):
        scores = self.cluster_assignment_distribution(x).items()
        labels, scores = zip(*scores)
        cid = np.random.choice(labels, p=scores)
        if cid == "new":
            self.clusters.append(component(x, prior = self.prior))
            self.n_cl += 1
        else:
            self.clusters[int(cid)] = self.add_datapoint_to_component(x, self.clusters[int(cid)])
        return
    
    @probit
    def add_new_point(self, x):
        self.n_pts += 1
        self.assign_to_cluster(x)
        self.update_alpha()
    
    def normalise_mixture(self):
        self.normalised = True
        for ss in self.clusters:
            ss.w = ss.N/self.n_pts
        self.clusters = np.array(self.clusters)
    
    @from_probit
    def sample_from_dpgmm(self, mixture, n_samps):
        if not self.normalised:
            self.normalise_mixture
        samples = np.array([mn(mixture[i].mu, mixture[i].sigma).rvs() for i in np.random.choice(np.arange(len(w)), p = w, size = n_samps)])
        return samples

if __name__ == '__main__':

    from tqdm import tqdm
    from scipy.stats import multivariate_normal as mn
    from corner import corner
    
    
#    samples = np.genfromtxt('/Users/stefanorinaldi/Desktop/posterior.dat', skip_header = 1, usecols = np.arange(11, dtype = int))
    samples = np.genfromtxt('data/GW150914_full_volume.txt')#, usecols = (0,1))
#    corner(samples)
#    plt.show()
#    exit()
#    samples[:,3] = samples[:,3] - np.mean(samples[:,3])
    prior_pars = (1, np.cov(samples.T), 1, np.mean(samples)) # k, L, nu, mu
    alpha0 = 1
    n_samps = len(samples)#1000
    DPGMM = mixture(prior_pars, alpha0)
    all_samples = []
    for s in tqdm(samples):
        DPGMM.add_new_point(np.atleast_2d(s))
        all_samples.append(s)
    
    
    samps_mix = sample_from_dpgmm(DPGMM.clusters, n_samps)
#    x  = np.linspace(np.min(samples),np.max(samples),1000)
#    dx = x[1]-x[0]
#    p  = np.zeros(len(x))
#    for comp in DPGMM.clusters:
#        p = p + comp.N*mn(comp.mu, comp.sigma).pdf(x)/n_samps
    c = corner(samples, color = 'red', hist_kwargs={'density':True})
    c = corner(samps_mix, fig = c, color = 'blue', hist_kwargs={'density':True})
    
    plt.show()


    
    
    
