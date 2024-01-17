from figaro.transform import transform_from_probit, transform_to_probit

"""
A gentle introduction to decorators: https://towardsdatascience.com/how-to-use-decorators-in-python-by-example-b398328163b
"""

def antiprobit(func):
    """
    Transform a point x from probit space to natural space and returns the function evaluated at the natural point y
    """
    def f_transf(ref, x, *args, **kwargs):
        if not ref.probit:
            return func(ref, x, *args)
        y = transform_from_probit(x, ref.bounds)
        return func(ref, y, *args)
    return f_transf

def probit(func):
    """
    Transform a point x from natural space to probit space and returns the function evaluated at the probit point y
    """
    def f_transf(ref, x, *args, **kwargs):
        if not ref.probit:
            return func(ref, x, *args)
        y = transform_to_probit(x, ref.bounds)
        return func(ref, y, *args)
    return f_transf

def from_probit(func):
    """
    Evaluate a function that samples points in probit space and return these points after transforming them to natural space
    """
    def f_transf(ref, *args, **kwargs):
        if not ref.probit:
            return func(ref, *args)
        y = func(ref, *args)
        return transform_from_probit(y, ref.bounds)
    return f_transf
