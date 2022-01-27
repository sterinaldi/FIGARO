from online_skyloc.transform import transform_from_probit, transform_to_probit

def antiprobit(func):
    def f_transf(ref, x, *args):
        y = transform_from_probit(x, ref.bounds)
        return func(y, *args)
    return f_transf

def probit(func):
    def f_transf(ref, x, *args):
        y = transform_to_probit(x, ref.bounds)
        return func(y, *args)
    return f_transf

def from_probit(func):
    def f_transf(ref, *args):
        y = func(*args)
        return transform_from_probit(y, ref.bounds)
    return f_transf
