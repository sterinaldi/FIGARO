# FIGARO - Fast Inference for GW Astronomy, Research & Observations

https://www.youtube.com/watch?v=uJeJ4YiVFz8

To install FIGARO, run `python setup.py install` and `python setup.py build_ext --inplace`.   
An introductive guide on how to use FIGARO can be found in the `introductive_guide.ipynb` notebook.

We recommend using the `igwn-py39` conda environment, which includes all the required packages apart from ImageIO.
This environment is available at https://computing.docs.ligo.org/conda/environments/igwn-py39/   
If you decide not to use `igwn-py39`, please remember that in order to have access to all the functions, LALSuite is required.
Without LALSuite, the following FIGARO functions won't be available:
* `figaro.load` module won't be able to load GW posterior samples and will raise an exception;
* `figaro.threeDvolume.VolumeReconstruction` will ignore any provided galaxy catalog. The volume reconstruction will be available.

In order to install LALSuite, follow the instructions provided in https://wiki.ligo.org/Computing/LALSuiteInstall

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6515977.svg)](https://doi.org/10.5281/zenodo.6515977)
