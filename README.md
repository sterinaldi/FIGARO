# FIGARO - Fast Inference for GW Astronomy, Research & Observations

https://www.youtube.com/watch?v=uJeJ4YiVFz8

To install FIGARO, run `python setup.py build_ext --inplace` and `python setup.py install`.

An introductive guide on how to use FIGARO can be found in the `introductive_guide.ipynb` notebook, where it is shown how to to reconstruct a probability density with FIGARO and how to use its products.\
To learn how to use FIGARO to reconstruct skymaps, have a look to the `skymaps.ipynb` notebook.

FIGARO comes with several plug-and-play console scripts:
* `figaro-density` reconstructs a probability density given a set of samples;
* `figaro-hierarchical` reconstructs a probability density given a set of single-event samples, each of them drawn around a sample from the initial probability density;
* `figaro-pp_plot` produces the so-called *pp-plots* for a set of single-event posterior samples to assess the validity of a simulated dataset;
* `figaro-mockdata` generates a set of synthetic posterior samples from a given hierarchical distribution.

In order to see the available options, run `console_script_name -h`.

We recommend using the `igwn-py39` conda environment, which includes all the required packages apart from ImageIO.
This environment is available at https://computing.docs.ligo.org/conda/environments/igwn-py39/   
If you decide not to use `igwn-py39`, please remember that in order to have access to all the functions, LALSuite is required.
Without LALSuite, the following FIGARO functions won't be available:
* `figaro.load` module won't be able to load GW posterior samples and will raise an exception;
* `figaro.threeDvolume.VolumeReconstruction` will ignore any provided galaxy catalog. The volume reconstruction will be available.

In order to install LALSuite, follow the instructions provided in https://wiki.ligo.org/Computing/LALSuiteInstall

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6515977.svg)](https://doi.org/10.5281/zenodo.6515977)
