from matplotlib import rcParams
from matplotlib import pyplot as plt
from shutil import which
import inspect

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

tex_flag = False
if which('latex') is not None:
    tex_flag = True
    rcParams["text.usetex"] = True
rcParams["xtick.labelsize"] = 14
rcParams["ytick.labelsize"] = 14
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"
rcParams["legend.fontsize"] = 12
rcParams["legend.frameon"]  = False
rcParams["legend.loc"]      = "best"
rcParams["axes.labelsize"]  = 16
rcParams["axes.grid"]       = True
rcParams["grid.alpha"]      = 0.6
rcParams["grid.linestyle"]  = "dotted"
rcParams["lines.linewidth"] = 0.7
rcParams["hist.bins"]       = "sqrt"
rcParams["savefig.bbox"]    = "tight"
rcParams["contour.negative_linestyle"] = "solid"

def nicer_hist(func):
    def decorated_func(*args, **kwargs):
        if not 'density' in kwargs.keys():
            kwargs['density'] = True
        if not 'histtype' in kwargs.keys():
            kwargs['histtype'] = 'step'
        return func(*args, **kwargs)
    return decorated_func

plt.hist = nicer_hist(plt.hist)
