import numpy as np
from figaro.exceptions import FIGAROException

def MC_integral(p, q, n_draws = 1e3, error = True, do_checks = True):
    """
    Monte Carlo integration using FIGARO reconstructions.
        ∫p(x)q(x)dx ~ ∑p(x_i)/N with x_i ~ q(x)
    
    p(x) must have a pdf() method and q(x) must have a rvs() method.
    Lists of p and q are also accepted.
    
    Arguments:
        :list or class instance p: the probability density to evaluate. Must have a pdf() method.
        :list or class instance q: the probability density to sample from. Must have a rvs() method.
        :int n_draws:              number of MC draws
        :bool error:               whether to return the uncertainty on the integral value or not.
        :bool do_checks:           whether to perform consistency checks or not (faster code, lazy evaluation - see https://docs.python.org/3/reference/expressions.html#boolean-operations).
    
    Return:
        :double: integral value
        :double: uncertainty (if error = True)
    """
    # Check that both p and q are iterables or callables:
    if do_checks and not ((callable(p) or np.iterable(p)) and (callable(q) or np.iterable(q))):
        raise FIGAROException("p and q must be callables or list of callables")
    # Number of p draws and methods check
    if np.iterable(p):
        if do_checks and not np.alltrue([hasattr(pi, 'pdf') for pi in p]):
            raise FIGAROException("p must have a pdf method")
        n_p = len(p)
        iter_p = True
    else:
        if do_checks and not hasattr(p, 'pdf'):
            raise FIGAROException("p must have a pdf method")
        iter_p = False
    # Number of q draws and methods check
    if np.iterable(q):
        if do_checks and not np.alltrue([hasattr(qi, 'rvs') for qi in q]):
            raise FIGAROException("q must have a rvs method")
        n_q = len(q)
        iter_q = True
    else:
        if do_checks and not hasattr(q, 'rvs'):
            raise FIGAROException("q must have a rvs method")
        iter_q = False

    n_draws = int(n_draws)

    # Integrals
    if iter_p and iter_q:
        shortest = np.min([n_p, n_q])
        probabilities = np.array([qi.pdf(pi.rvs(n_draws)) for pi, qi in zip(p[:shortest], q[:shortest])])
    elif iter_p and not iter_q:
        probabilities = np.array([q.pdf(pi.rvs(n_draws)) for pi in p])
    elif iter_q and not iter_p:
        samples = p.rvs(n_draws)
        probabilities = np.array([qi.pdf(samples) for qi in q])
    else:
        probabilities = np.atleast_1d(q.pdf(p.rvs(n_draws)))
    
    means = probabilities.mean(axis = 1)
    I = means.mean()
    if not error:
        return I
    mc_error = (probabilities.var(axis = 1)/n_draws).mean()
    figaro_error = means.var()
    return I, np.sqrt(mc_error + figaro_error)
