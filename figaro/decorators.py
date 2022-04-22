from figaro.transform import transform_from_probit, transform_to_probit, probit_logJ
from figaro.coordinates import celestial_to_cartesian, cartesian_to_celestial
import numpy as np

"""
A gentle introduction to decorators: https://towardsdatascience.com/how-to-use-decorators-in-python-by-example-b398328163b
"""

def antiprobit(func):
    """
    Transform a point x from probit space to natural space and returns the function evaluated at the natural point y
    """
    def f_transf(ref, x, *args, **kwargs):
        y = transform_from_probit(x, ref.bounds)
        return func(ref, y, *args)
    return f_transf

def probit(func):
    """
    Transform a point x from natural space to probit space and returns the function evaluated at the probit point y
    """
    def f_transf(ref, x, *args, **kwargs):
        y = transform_to_probit(x, ref.bounds)
        return func(ref, y, *args)
    return f_transf

def from_probit(func):
    """
    Evaluate a function that samples points in probit space and return these points after transforming them to natural space
    """
    def f_transf(ref, *args, **kwargs):
        y = func(ref, *args)
        return transform_from_probit(y, ref.bounds)
    return f_transf

def jacobian_probit(func):
    """
    Evaluates a function and returns it multiplied by the jacobian of the probit transformation
    """
    def f_transf(ref, x, *args, **kwargs):
        y = func(ref, x)
        return y*np.exp(probit_logJ(x, ref.bounds))
    return f_transf

def jacobian_log_probit(func):
    """
    Evaluates a function and returns the logarithm of it multiplied by the jacobian of the probit transformation
    """
    def f_transf(ref, x, *args, **kwargs):
        y = func(ref, x)
        return y + probit_logJ(x, ref.bounds)
    return f_transf

def inv_jacobian_probit(func):
    """
    Evaluates a function and returns it multiplied by the inverse of the jacobian of the probit transformation
    """
    def f_transf(ref, x, *args, **kwargs):
        y = func(ref, x)
        return y*np.exp(-probit_logJ(x, ref.bounds))
    return f_transf

def inv_jacobian_log_probit(func):
    """
    Evaluates a function and returns the logarithm of it multiplied by the jacobian of the probit transformation
    """
    def f_transf(ref, x, *args, **kwargs):
        y = func(ref, x)
        return y - probit_logJ(x, ref.bounds)
    return f_transf

def cartesian(func):
    """
    Transform a point x from celestial coordinates to cartesian coordinates and returns the function evaluated at the probit point y
    """
    def f_transf(ref, x, *args, **kwargs):
        y = celestial_to_cartesian(x)
        return func(ref, y, *args)
    return f_transf

def celestial(func):
    """
    Transform a point x from cartesian coordinates to celestial coordinates and returns the function evaluated at the probit point y
    """
    def f_transf(ref, x, *args, **kwargs):
        y = cartesian_to_celestial(x)
        return func(y, *args)
    return f_transf
