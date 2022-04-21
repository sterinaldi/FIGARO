# FIGARO
## Fast Inference for GW Astronomy, Research & Observations

https://www.youtube.com/watch?v=uJeJ4YiVFz8

To install the package, run `pip install -r requirements.txt`, `python setup.py install` and finally `python setup.py build_ext --inplace`.

Please remember that in order to have access to all the functions, LALSuite is required.
In order to install LALSuite, follow the instructions provided in https://wiki.ligo.org/Computing/LALSuiteInstall  
We recommend using the igwn-py39 conda environment, which includes all the required packages apart from ImageIO.
This environment is available at https://computing.docs.ligo.org/conda/environments/igwn-py39/ 

An introductive guide on how to use FIGARO can be found in the `introductive_guide.ipynb` notebook.
