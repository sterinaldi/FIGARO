# [FIGARO - Fast Inference for GW Astronomy, Research & Observations](https://www.youtube.com/watch?v=uJeJ4YiVFz8)

FIGARO is an inference code designed to estimate multivariate probability densities given samples from an unknown distribution using a Dirichlet Process Gaussian Mixture Model (DPGMM) as nonparameteric model.
It is also possible to perform hierarchical inferences: in this case, the model used is (H)DPGMM, described in [Rinaldi & Del Pozzo (2022a)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.5454R/abstract).
Differently from other DPGMM implementations relying on variational algorithms, FIGARO does not require the user to specify a priori the maximum allowed number of mixture components. The required number of Gaussian distributions to be included in the mixture is inferred from the data.

An introductive guide on how to use FIGARO can be found in the `introductive_guide.ipynb` notebook, where it is shown how to to reconstruct a probability density with FIGARO and how to use its products.
We strongly encourage the interested user to go through the whole notebook, since it provides a (hopefully detailed) tutorial on how to properly set and use FIGARO.\
To learn how to use FIGARO to reconstruct skymaps, have a look at the `skymaps.ipynb` notebook. In that notebook we show how to obtain the skymaps included in [Rinaldi & Del Pozzo (2022b)](https://ui.adsabs.harvard.edu/abs/2022arXiv220507252R/abstract) - please cite this paper if you use FIGARO in your research.

You can install FIGARO by running `source install.sh`: this script will go through all the necessary steps to install the code.
We recommend using one of the following two conda environments:
* `figaro_env` is a dedicated environment already containing all the required packages. It is created by the installer if the option `-e` is provided;
* `igwn-py39`, which includes all the required packages apart from ImageIO, is available [here](https://computing.docs.ligo.org/conda/environments/igwn-py39) .

If you prefer install FIGARO by hand, run `python setup.py build_ext --inplace` and `python setup.py install`. In some cases (like on clusters), it may happen that you do not have the permission to write in the default installation directory. In this case, run `python setup.py install --user`.

If you decide not to use one of the default environments, please remember that in order to have access to all the functions, LALSuite is required.
Without LALSuite, the following FIGARO functions won't be available:
* `figaro.load` module won't be able to load GW posterior samples and will raise an exception;
* `figaro.threeDvolume.VolumeReconstruction` will ignore any provided galaxy catalog. The volume reconstruction will be available.

To install LALSuite, follow the instructions provided [here](https://wiki.ligo.org/Computing/LALSuiteInstall). In most cases, `conda install -c conda-forge lalsuite` will work.
The parallelized scripts use [Ray](https://docs.ray.io/en/latest/) to parallelize. Ray is not included in `igwn-py39` nor is automatically installed with FIGARO (but it is included in `figaro_env`). 
If you wish to use these scripts, please install Ray via `pip install ray` and then (re-)install FIGARO to include also the parallelized scripts.

FIGARO comes with several plug-and-play console scripts:
* `figaro-density` reconstructs a probability density given a set of samples;
* `figaro-hierarchical` reconstructs a probability density given a set of single-event samples, each of them drawn around a sample from the initial probability density;
* `figaro-pp_plot` produces the so-called *pp-plots* for a set of single-event posterior samples to assess the validity of a simulated dataset;
* `figaro-mockdata` generates a set of synthetic posterior samples from a given hierarchical distribution;
* `figaro-entropy` reconstruct a probability density and provides an estimate of the entropy as a function of the number of samples.

In addition to these, the parallelized version of the inference scripts are available:
* `figaro-par-density`, parallelized sampling;
* `figaro-par-hierarchical`, parallelized single-event analysis and parallelized sampling.

In order to see the available options, run `console_script_name -h`.
