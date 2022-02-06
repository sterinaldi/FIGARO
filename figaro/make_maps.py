import plotly.graph_objects as go
import numpy as np
import sys
from itertools import product
from figaro.cumulative import fast_log_cumulative
from figaro.credible_regions import FindHeights
import plotly.graph_objects as go #conda install -c plotly plotly=5.5.0

def volume_rendering(VRC, dmax = 600):
    # Create a cartesian grid
    N = 64
    eps = 1
    MAX = dmax-eps
    x = np.linspace(-MAX,MAX,N)
    y = np.linspace(-MAX,MAX,N)
    z = np.linspace(-MAX,MAX,N)
    X, Y, Z = np.meshgrid(x,y,z)
    sys.stderr.write("producing 3 dimensional map\n")
    grid = np.array([v for v in product(*(x,y,z))])
    
    log_cartesian_map = VRC.evaluate_log_mixture(grid)
    
    # create a normalized cumulative distribution
    log_cartesian_sorted = np.ascontiguousarray(np.sort(log_cartesian_map.flatten())[::-1])
    log_cartesian_cum = fast_log_cumulative(log_cartesian_sorted)
    # find the indeces corresponding to the given CLs
    adLevels = np.ravel([0.05,0.5,0.95])
    args = [(log_cartesian_sorted,log_cartesian_cum,level) for level in adLevels]
    adHeights = [FindHeights(a) for a in args]
    heights = {str(lev):hei for lev,hei in zip(adLevels,adHeights)}
    fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=log_cartesian_map.flatten(),
    isomin=heights['0.95'],
    isomax=np.max(log_cartesian_map),
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=10, # needs to be a large number for good volume rendering
    colorscale='RdBu'
    ))
    fig.write_html('tmp.html', auto_open=True)
