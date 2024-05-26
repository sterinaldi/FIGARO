# [FIGARO - Fast Inference for GW Astronomy, Research & Observations](https://www.youtube.com/watch?v=uJeJ4YiVFz8)

FIGARO is an inference code designed to estimate multivariate probability densities given samples from an unknown distribution using a Dirichlet Process Gaussian Mixture Model (DPGMM) as nonparameteric model.
It is also possible to perform hierarchical inferences: in this case, the model used is (H)DPGMM, described in [Rinaldi & Del Pozzo (2022a)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.5454R/abstract).
Differently from other DPGMM implementations relying on variational algorithms, FIGARO does not require the user to specify a priori the maximum allowed number of mixture components. The required number of Gaussian distributions to be included in the mixture is inferred from the data. The documentation and user guide for FIGARO is available at [the documentation page](https://figaro.readthedocs.io).

[![DOI](https://joss.theoj.org/papers/10.21105/joss.06589/status.svg)](https://doi.org/10.21105/joss.06589)
![Test](https://github.com/sterinaldi/FIGARO/actions/workflows/test.yml/badge.svg)
## Getting started

You can install FIGARO either via pip (stable release, recommended) 
```
pip install figaro
```
or from the repository (potentially unstable)
```
git clone git@github.com:sterinaldi/FIGARO.git
cd FIGARO
pip install .
```

FIGARO comes with two plug-and-play CLI:

* `figaro-density` reconstructs a probability density given a set of samples;
* `figaro-hierarchical` reconstructs a probability density given a set of single-event samples, each of them drawn around a sample from the initial probability density.

If you only want to reconstruct some probability density or run a vanilla hierarchical analysis, we strongly recommend using these CLI, which are already tested and optimised. A (hopefully gentle) introduction to them can be found at [this page](https://figaro.readthedocs.io/en/latest/quickstart.html), and a guide on how to use the FIGARO reconstructions is available [here](https://figaro.readthedocs.io/en/latest/use_mixture.html).
If you want to include FIGARO in your own scripts, an introductive guide can be found [here](https://figaro.readthedocs.io/en/latest/python_script.html): there we show how to to reconstruct a probability density with FIGARO and how to use its products.

## Acknowledgments

If you use FIGARO in your research, please cite [Rinaldi & Del Pozzo (2024)](https://joss.theoj.org/papers/10.21105/joss.06589):
```text
@ARTICLE{Rinaldi2024,
       author = {{Rinaldi}, Stefano and {Del Pozzo}, Walter},
        title = "{FIGARO: hierarchical non-parametric inference for population studies}",
      journal = {Journal of Open Source Software},
    publisher = {The Open Journal},
         year = 2024,
        month = may,
       volume = {9},
       number = {97},
        pages = {6589},
          doi = {10.21105/joss.06589},
          url = {https://doi.org/10.21105/joss.06589}
}
```

If you make use of the hierarchical analysis, you should mention (H)DPGMM as the model used and cite [Rinaldi & Del Pozzo (2022)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.5454R/abstract):

```text
@ARTICLE{2022MNRAS.509.5454R,
       author = {{Rinaldi}, Stefano and {Del Pozzo}, Walter},
        title = "{(H)DPGMM: a hierarchy of Dirichlet process Gaussian mixture models for the inference of the black hole mass function}",
      journal = {\mnras},
     keywords = {gravitational waves, methods: data analysis, methods: statistical, stars: black holes, Astrophysics - Instrumentation and Methods for Astrophysics, General Relativity and Quantum Cosmology},
         year = 2022,
        month = feb,
       volume = {509},
       number = {4},
        pages = {5454-5466},
          doi = {10.1093/mnras/stab3224},
archivePrefix = {arXiv},
       eprint = {2109.05960},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.5454R},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

