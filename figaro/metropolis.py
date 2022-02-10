import numpy as np
from figaro.integral import log_integrand_1d
from numpy.random import uniform

def propose_point(old_point, dm, ds):
    m = old_point[0] + uniform(-1,1)*dm
    s = old_point[1] + uniform(-1,1)*ds
    return [m,s]

def sample_point(events, m_min = -20, m_max = 20, s_min = 0, s_max = 1, burnin = 1000, dm = 1, ds = 0.05, a = 1, b = 0.2):
    old_point = [uniform(m_min, m_max), uniform(s_min, s_max)]
    for _ in range(burnin):
        new_point = propose_point(old_point, dm, ds)
        if not (s_min < new_point[1] < s_max and m_min < new_point[0] < m_max):
            log_new = -np.inf
            log_old = 0.
        else:
            log_new = log_integrand_1d(new_point[0], new_point[1], events, a, b)
            log_old = log_integrand_1d(old_point[0], old_point[1], events, a, b)
        if log_new - log_old > np.log(uniform()):
            old_point = new_point
    return np.array([old_point[0], old_point[1]])
