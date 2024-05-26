.. FIGARO documentation master file, created by
   sphinx-quickstart on Sat Jul 15 21:24:28 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FIGARO - Fast Inference for GW Astronomy, Research & Observations
=================================================================

| This page contains the documentation for `FIGARO <https://github.com/sterinaldi/FIGARO>`_, along with a brief description of how to use it in your research project. Please refer to the `GitHub issue tracker <https://github.com/sterinaldi/FIGARO/issues>`_ for bug reports (extremely appreciated), issues and contributions.
| If you're curious about the other projects I'm working on or if you want to get in touch with me, feel free to visit `my website <https://sterinaldi.github.io>`_!

.. image:: https://joss.theoj.org/papers/10.21105/joss.06589/status.svg
   :target: https://doi.org/10.21105/joss.06589
   
Statement of need
-----------------
| FIGARO is an inference code designed to estimate multivariate probability densities given samples from an unknown distribution using a Dirichlet Process Gaussian Mixture Model (DPGMM) as nonparameteric model, and to perform hierarchical non-parametric inferences: in this case, the model used is (H)DPGMM, described in `Rinaldi & Del Pozzo (2022a) <https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.5454R/abstract>`_.
| This code, originally developed in the context of black hole population studies using gravitational-wave observations, take as input generic data and therefore it can be applied to a variety of studies beyond gravitational wave populations.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation.md
   quickstart.md
   use_mixture.ipynb
   python_script.ipynb
   acknowledgment.md
   api
   figaro
   

Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
