from online_skyloc.transform import transform_from_probit, transform_to_probit, probit_logJ
from online_skyloc.coordinates import celestial_to_cartesian, cartesian_to_celestial
import numpy as np

def antiprobit(func):
    def f_transf(ref, x, *args):
        y = transform_from_probit(x, ref.bounds)
        return func(ref, y, *args)
    return f_transf

def probit(func):
    def f_transf(ref, x, *args):
        y = transform_to_probit(x, ref.bounds)
        return func(ref, y, *args)
    return f_transf

def from_probit(func):
    def f_transf(ref, *args):
        y = func(ref, *args)
        return transform_from_probit(y, ref.bounds)
    return f_transf

def jacobian_probit(func):
    def f_transf(ref, x, *args):
        y = func(ref, x)
        return y*np.exp(probit_logJ(x, ref.bounds))
    return f_transf

def cartesian(func):
    def f_transf(ref, x, *args):
        y = celestial_to_cartesian(x)
        return func(ref, y, *args)
    return f_transf

def celestial(func):
    def f_transf(ref, x, *args):
        y = cartesian_to_celestial(x)
        return func(y, *args)
    return f_transf
