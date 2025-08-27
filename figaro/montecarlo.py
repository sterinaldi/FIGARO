import numpy as np
from figaro.exceptions import FIGAROException

def MC_integral(p, q, n_draws = 1e4, error = True):
    """
    Monte Carlo integration using FIGARO reconstructions.
        ∫p(x)q(x)dx ~ ∑p(x_i)/N with x_i ~ q(x)
    
    p(x) must have a pdf() method and q(x) must have a rvs() method.
    Lists of p and q are also accepted.
    
    Arguments:
        list or class instance p: the probability density to evaluate. Must be callable or have a pdf() method.
        list or class instance q: the probability density to sample from. Must have a rvs() method.
        int n_draws:              number of MC draws
        bool error:               whether to return the uncertainty on the integral value or not.
    
    Return:
        double: integral value
        double: uncertainty (if error = True)
    """
    # Check that both p and q are iterables or callables:
    if not ((hasattr(p, '__call__') or hasattr(p, 'pdf') or np.iterable(p)) and (hasattr(q, 'rvs') or np.iterable(q))):
        raise FIGAROException("p and q must be list of callables or having pdf/rvs methods")
    # Number of p draws and methods check
    iter_p = False
    iter_q = False
    if np.iterable(p):
        if not np.alltrue([(hasattr(pi, '__call__') or hasattr(pi, 'pdf')) for pi in p]):
            raise FIGAROException("p must be callable or have pdf method")
        n_p = len(p)
        np.random.shuffle(p)
        iter_p = True
    else:
        if not (hasattr(p, '__call__') or hasattr(p, 'pdf')):
            raise FIGAROException("p must be callable or have pdf method")
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
        try:
            probabilities = np.array([pi(qi.rvs(n_draws)) for pi, qi in zip(p[:shortest], q[:shortest])])
        except:
            probabilities = np.array([pi.pdf(qi.rvs(n_draws)) for pi, qi in zip(p[:shortest], q[:shortest])])
    elif iter_q and not iter_p:
        try:
            probabilities = np.array([p(qi.rvs(n_draws)) for qi in q])
        except:
            probabilities = np.array([p.pdf(qi.rvs(n_draws)) for qi in q])
    elif iter_p and not iter_q:
        samples = q.rvs(n_draws)
        try:
            probabilities = np.array([pi(samples) for pi in p])
        except:
            probabilities = np.array([pi.pdf(samples) for pi in p])
    else:
        try:
            probabilities = np.atleast_2d(p(q.rvs(n_draws)))
        except:
            probabilities = np.atleast_2d(p.pdf(q.rvs(n_draws)))
    means = probabilities.mean(axis = 1)
    I = means.mean()
    if not error:
        return I
    mc_error = (probabilities.var(axis = 1)/n_draws).mean()
    figaro_error = means.var()/len(means)
    return I, np.sqrt(mc_error + figaro_error)

def KL_divergence(p, q, n_draws = 1e4, base = 'e'):
    if np.iterable(p) and np.iterable(q):
        return np.array([_KL_divergence(pi, qi, n_draws = n_draws, base = base) for pi, qi in zip(p, q)])
    elif np.iterable(p):
        return np.array([_KL_divergence(pi, q, n_draws = n_draws, base = base) for pi in p])
    elif np.iterable(q):
        return np.array([_KL_divergence(p, qi, n_draws = n_draws, base = base) for qi in q])
    else:
        return _KL_divergence(p, q, n_draws = n_draws, base = base)

log_dict = {'e': np.log, '10': np.log10, '2': np.log2}

def _KL_divergence(p, q, n_draws = 1e4, base = 'e'):
    log = log_dict[str(base)]
    if hasattr(p, 'logpdf') and hasattr(q, 'logpdf'):
        R = lambda x: (p.logpdf(x)-q.logpdf(x))*log(np.e)
    else:
        if hasattr(p, 'pdf'):
            p_pdf = p.pdf
        else:
            p_pdf = p
        if hasattr(q, 'pdf'):
            q_pdf = q.pdf
        else:
            q_pdf = q
        R = lambda x: log(p_pdf(x))-log(q_pdf(x))
    return MC_integral(R, p, n_draws = n_draws, error = False)

def JS_distance(p, q, n_draws = 1e4, base = 'e'):
    if np.iterable(p) and np.iterable(q):
        return np.array([_JS_distance(pi, qi, n_draws = n_draws, base = base) for pi, qi in zip(p, q)])
    elif np.iterable(p):
        return np.array([_JS_distance(pi, q, n_draws = n_draws, base = base) for pi in p])
    elif np.iterable(q):
        return np.array([_JS_distance(p, qi, n_draws = n_draws, base = base) for qi in q])
    else:
        return _JS_distance(p, q, n_draws = n_draws, base = base)

def _JS_distance(p, q, n_draws = 1e4, base = 'e'):
    if hasattr(p, 'pdf') and hasattr(q, 'pdf'):
        m = lambda x: 0.5*(p.pdf(x) + q.pdf(x))
    else:
        m = lambda x: 0.5*(p(x) + q(x))
    return np.sqrt(0.5*(_KL_divergence(p, m) + _KL_divergence(q, m)))
