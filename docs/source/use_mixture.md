# How to use a FIGARO reconstruction

FIGARO, despite being a stochastic sampler, does not produce a set of samples (numbers) directly, but rather a number of `figaro.mixture.mixture` objects stored as `.json` or `.pkl` files. Each of these objects represents a probability density drawn around the probability density that generated the available data. If you need to use these realisations in a `python` script (to evaluate their pdf, for example, or to draw realisations from them), here you'll find the (few!) steps needed for doing so.


## Basic usage 
Let us assume to have our realisations stored in a file called `draws_file.json` (they can be the product of a hierarchical inference or a DPGMM reconstruction, it does not make any difference). The first thing to do is to load the realisations via the dedicated method, `figaro.load.load_density`:

```python
from figaro.load import load_density
draws = load_density('./draws_file.json')
```

`draws` will be a list of `figaro.mixture.mixture` objects, each of them representing a probability distribution. The methods of this class are modelled after the `scipy.stats` methods to facilitate users that are already familiar with the SciPy package:

```python
d       = draws[0]
X       = [x, y, z]
p       = d.pdf(X)
log_p   = d.logpdf(X)
c       = d.cdf(X)
samples = d.rvs(size = 100)
```

## Advanced use

If you need to draw samples from the median distribution, there is a dedicated method in the `figaro.utils` module:

```python
from figaro.utils import rvs_median
samples = rvs_median(draws)
```

The gradient of the recovered distribution can be evaluated using the `gradient()` method of individual draws:
```python
g = d.gradient(X)
```
It is also possible to evaluate the gradient of the median distribution in a numerically stable way using the dedicated function:
```python
from figaro.utils import gradient_median
g_median = gradient_median(X, draws)
```
Please note: these methods are painfully slow and we haven't really found a way of optimising them. If you manage to improve it, please send us a pull request!

For multivariate distributions, it might happen that one needs to evaluate the conditional distribution or the marginal distribution.
Making use of the properties of the multivariate Gaussian distribution, we can obtain the conditional and/or marginal distribution analytically both via the methods included in the `figaro.mixture.mixture` class or via the ones in the `figaro.marginal` module:

```python
from figaro.marginal import condition, marginalise

# Marginalisation over the last two dimensions of a 3D reconstruction
marg_draws = [d.marginalise((1,2)) for d in draws]
marg_draws = marginalise(draws, (1,2))

# Condition on a specific value Y = [y'] of the second dimension
cond_draws = [d.condition(Y, 2) for d in draws]
cond_draws = condition(draws, Y, 2)
```

Please note that in both cases the original `draws` list is preserved.


## Plots
The plots produced by the CLI can be easily reproduced using the methods included in the `figaro.plot` module. Please refer to its documentation page for the details. 
