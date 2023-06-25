from matplotlib import rcParams
from matplotlib.pyplot import hist
from distutils.spawn import find_executable

tex_flag = False
if find_executable('latex'):
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

# Better way of doing this?
histdefaults = list(hist.__defaults__)
histdefaults[2] = True   # density
histdefaults[6] = 'step' # histtype
hist.__defaults__ = tuple(histdefaults)
