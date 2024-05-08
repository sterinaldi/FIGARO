# Contribution guide

Thanks for your interest in FIGARO! This project is an open source project released under the MIT license and welcomes contributions in the form of bug reports, feature requests and pull requests. The code is hosted on [GitHub](https://github.com/sterinaldi/FIGARO), the documentation is available on [readthedocs](https://figaro.readthedocs.io).

## Bug report
Please report bugs on the [Issue tracker](https://github.com/sterinaldi/FIGARO/issues).
When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case, and/or steps to reproduce the issue. In particular, please include a [Minimal, Reproducible Example](https://stackoverflow.com/help/minimal-reproducible-example).

For some specific applications (GW and cosmology), FIGARO makes use of [LAL](https://lscsoft.docs.ligo.org/lalsuite/lalsuite/index.html). If you encounter a related issue, please check that the issue is not due to your LAL installation before reporting a bug.

## New features and pull requests
New features can be requested and discussed in the [Issue tracker](https://github.com/sterinaldi/FIGARO/issues) and [pull requests](https://github.com/sterinaldi/FIGARO/pulls) are welcome. While requesting a feature or creating a pull request, please keep in mind that the idea is to keep the inputs for FIGARO as generic as possible (with the notable exception of GW posterior samples files). You are kindly asked not to propose/request features aimed at loading data from highly structured files.

### Acknowledgments
This guide is based on [@nayafia](https://github.com/nayafia)'s [contributing template](https://github.com/nayafia/contributing-template) and [@dfm](https://github.com/dfm)'s [corner contributing guide](https://github.com/dfm/corner.py/CONTRIBUTING.md).
