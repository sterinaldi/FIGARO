# How to use the FIGARO reconstructions

FIGARO, despite being a stochastic sampler, does not produce a set of samples (numbers) directly, but rather a number of `figaro.mixture.mixture` objects stored as `.json` or `.pkl` files. Each of these objects represents a probability density drawn around the probability density that generated the available data.


