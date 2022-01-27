from .transform import transform_from_probit, transform_to_probit, transform_from_hypercube, transform_to_hypercube

def antiprobit(func):
    def f_transf(ref, x):
        y = transform_from_probit(x, ref.bounds)
        return func(y)
    return f_transf

def probit(func):
    def f_transf(ref, x):
        y = transform_to_probit(x, ref.bounds)
        return func(y)
    return f_transf

def to_hypercube(func):
    def f_transf(ref, x):
        y = transform_to_hypercube(x, ref.bounds)
        return func(y)
    return f_transf

def from_hypercube(func):
    def f_transf(ref, x):
        y = transform_from_hypercube(x, ref.bounds)
        return func(y)
    return f_transf
