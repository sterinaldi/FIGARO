import numpy as np
from figaro.exceptions import FIGAROException

def MC_integral(p, q, n_draws = 1e4, error = True):
    """
    Monte Carlo integration using FIGARO reconstructions.
        ∫p(x)q(x)dx ~ ∑p(x_i)/N with x_i ~ q(x)
    
    p(x) must have a pdf() method and q(x) must have a rvs() method.
    Lists of p and q are also accepted.
    
    Arguments:
        list or class instance p: the probability density to evaluate. Must have a pdf() method.
        list or class instance q: the probability density to sample from. Must have a rvs() method.
        int n_draws:              number of MC draws
        bool error:               whether to return the uncertainty on the integral value or not.
    
    Return:
        double: integral value
        double: uncertainty (if error = True)
    """
    # Check that both p and q are iterables or callables:
    if not ((hasattr(p, 'pdf') or np.iterable(p)) and (hasattr(q, 'rvs') or np.iterable(q))):
        raise FIGAROException("p and q must be list of callables or having pdf/rvs methods")
    # Number of p draws and methods check
    iter_p = False
    iter_q = False
    if np.iterable(p):
        if not np.alltrue([hasattr(pi, 'pdf') for pi in p]):
            raise FIGAROException("p must have pdf method")
        n_p = len(p)
        np.random.shuffle(p)
        iter_p = True
    else:
        if not hasattr(p, 'pdf'):
            raise FIGAROException("p must have pdf method")
    # Number of q draws and methods check
    if np.iterable(q):
        if not np.alltrue([hasattr(qi, 'rvs') for qi in q]):
            raise FIGAROException("q must have rvs method")
        n_q = len(q)
        np.random.shuffle(q)
        iter_q = True
    else:
        if not hasattr(q, 'rvs'):
            raise FIGAROException("q must have rvs method")

    n_draws = int(n_draws)
    # Integrals
    if iter_p and iter_q:
        shortest = np.min([n_p, n_q])
        probabilities = np.array([pi.pdf(qi.rvs(n_draws)) for pi, qi in zip(p[:shortest], q[:shortest])])
    elif iter_q and not iter_p:
        probabilities = np.array([p.pdf(qi.rvs(n_draws)) for qi in q])
    elif iter_p and not iter_q:
        samples = q.rvs(n_draws)
        probabilities = np.array([pi.pdf(samples) for pi in p])
    else:
        probabilities = np.atleast_2d(p.pdf(q.rvs(n_draws)))
    means = probabilities.mean(axis = 1)
    I = means.mean()
    if not error:
        return I
    mc_error = (probabilities.var(axis = 1)/n_draws).mean()
    figaro_error = means.var()/len(means)
    return I, np.sqrt(mc_error + figaro_error)
