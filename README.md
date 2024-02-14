# [FIGARO - Fast Inference for GW Astronomy, Research & Observations](https://www.youtube.com/watch?v=uJeJ4YiVFz8)

FIGARO is an inference code designed to estimate multivariate probability densities given samples from an unknown distribution using a Dirichlet Process Gaussian Mixture Model (DPGMM) as nonparameteric model.
It is also possible to perform hierarchical inferences: in this case, the model used is (H)DPGMM, described in [Rinaldi & Del Pozzo (2022a)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.5454R/abstract).
Differently from other DPGMM implementations relying on variational algorithms, FIGARO does not require the user to specify a priori the maximum allowed number of mixture components. The required number of Gaussian distributions to be included in the mixture is inferred from the data. The documentation and user guide for FIGARO is available at https://figaro.readthedocs.io .

## Getting started

You can install FIGARO either via pip (`pip install figaro`, stable release) or from source (clone this repository and then `pip install .`, possibily unstable).

FIGARO comes with two plug-and-play console scripts:
* `figaro-density` reconstructs a probability density given a set of samples;
* `figaro-hierarchical` reconstructs a probability density given a set of single-event samples, each of them drawn around a sample from the initial probability density.

In addition to these, the parallelized version of the inference scripts (`figaro-par-density` and `figaro-par-hierarchical`) are available (see below for a note on them). 
The basic usage for the serial scripts is:
* `figaro-density -i path/to/samples.txt -b "[[xmin,xmax]]`;
* `figaro-hierarchical -i path/to/folder -b "[[xmin,xmax]]` where `folder` stores the single-event samples files `samples_1.txt`, `samples_2.txt`, and so on. 

Options can also be provided via a config file: `console_script_name --config your_config_file.ini`. An example of config file can be found in `options_example.ini`.
In order to see all the available options, run `console_script_name -h`.

If you only want to reconstruct some probability density or run a vanilla hierarchical analysis, we strongly recommend using these scripts, which are already tested and optimised.
However, if you want to include FIGARO in your own scripts, an introductive guide can be found in the `introductive_guide.ipynb` notebook, where we show how to to reconstruct a probability density with FIGARO and how to use its products. We strongly encourage the interested user to go through the whole notebook, since it provides a (hopefully detailed) tutorial on how to properly set and use FIGARO.

## Acknowledgments

If you use FIGARO in your research, please cite [Rinaldi & Del Pozzo (2022b)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.517L...5R/abstract):
```
@ARTICLE{2022MNRAS.517L...5R,
       author = {{Rinaldi}, Stefano and {Del Pozzo}, Walter},
        title = "{Rapid localization of gravitational wave hosts with FIGARO}",
      journal = {\mnras},
     keywords = {gravitational waves, methods: data analysis, methods: statistical, Astrophysics - Instrumentation and Methods for Astrophysics, General Relativity and Quantum Cosmology},
         year = 2022,
        month = nov,
       volume = {517},
       number = {1},
        pages = {L5-L10},
          doi = {10.1093/mnrasl/slac101},
archivePrefix = {arXiv},
       eprint = {2205.07252},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022MNRAS.517L...5R},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

If you make use of the hierarchical analysis, you should mention (H)DPGMM as the model used and cite [Rinaldi & Del Pozzo (2022a)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.5454R/abstract):

```
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

