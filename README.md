# [FIGARO - Fast Inference for GW Astronomy, Research & Observations](https://www.youtube.com/watch?v=uJeJ4YiVFz8)

FIGARO is an inference code designed to estimate multivariate probability densities given samples from an unknown distribution using a Dirichlet Process Gaussian Mixture Model (DPGMM) as nonparameteric model.
It is also possible to perform hierarchical inferences: in this case, the model used is (H)DPGMM, described in [Rinaldi & Del Pozzo (2022a)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.5454R/abstract).
Differently from other DPGMM implementations relying on variational algorithms, FIGARO does not require the user to specify a priori the maximum allowed number of mixture components. The required number of Gaussian distributions to be included in the mixture is inferred from the data.
Please cite [Rinaldi & Del Pozzo (2022b)](https://ui.adsabs.harvard.edu/abs/2022arXiv220507252R/abstract) if you use FIGARO in your research. The documentation and user guide for FIGARO is available at https://figaro.readthedocs.io .

## Getting started

You can install FIGARO either via pip (`pip install figaro`, stable release) or from source (clone this repository and then `pip install .`, possibily unstable).

FIGARO comes with two plug-and-play console scripts:
* `figaro-density` reconstructs a probability density given a set of samples;
* `figaro-hierarchical` reconstructs a probability density given a set of single-event samples, each of them drawn around a sample from the initial probability density.

In addition to these, the parallelized version of the inference scripts (`figaro-par-density` and `figaro-par-hierarchical`) are available (see below for a note on them). 
The basic usage for the serial scripts is:
* `figaro-density -i path/to/samples.txt -b "[[xmin,xmax]]`;
* `figaro-hierarchical -i path/to/folder -b "[[xmin,xmax]]` where `folder` stores the single-event samples files `samples_1.txt`, `samples_2.txt`, and so on. 

In order to see all the available options, run `console_script_name -h`.

If you only wish to reconstruct some probability density or run a vanilla hierarchical analysis, we strongly recommend using these scripts, which are already tested and optimised.
However, if you wish to include FIGARO in your own scripts, an introductive guide can be found in the `introductive_guide.ipynb` notebook, where we show how to to reconstruct a probability density with FIGARO and how to use its products. We strongly encourage the interested user to go through the whole notebook, since it provides a (hopefully detailed) tutorial on how to properly set and use FIGARO.

