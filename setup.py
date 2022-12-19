import numpy
from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from codecs import open
from os import path
from distutils.extension import Extension
import os
import warnings

lal_flag = True
try:
    import lal
except ModuleNotFoundError:
    lal_flag = False

ray_flag = True
try:
    import ray
except ModuleNotFoundError:
    ray_flag = False

try:
    from Cython.Build import cythonize
except ImportError:
    raise ImportError("Cython not found. Please install it via\n\tpip install Cython")

if os.environ['CONDA_DEFAULT_ENV'] == 'igwn-py39':
    requirements = ['imageio']
else:
    with open("requirements.txt") as requires_file:
        requirements = requires_file.read().split("\n")

with open("README.md") as readme_file:
    long_description = readme_file.read()

# see https://stackoverflow.com/a/21621689/1862861 for why this is here
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        self.include_dirs.append(numpy.get_include())

ext_modules=[
             Extension("figaro.cumulative",
                       sources=[os.path.join("figaro","cumulative.pyx")],
                       libraries=["m"], # Unix-like specific
                       extra_compile_args=["-O3","-ffast-math"],
                       include_dirs=['figaro', numpy.get_include()]
                       ),
            ]
if lal_flag:
    if "LAL_PREFIX" in os.environ:
        # Older LAL installations requires this
        lal_prefix     = os.environ.get("LAL_PREFIX")
        lal_includes   = lal_prefix+"/include"
        lal_libs       = lal_prefix+"/lib"
        ext_modules.append(Extension("figaro.cosmology",
                          sources=[os.path.join("figaro","cosmology.pyx")],
                          libraries=["m", "lal"], # Unix-like specific
                          library_dirs = [lal_libs],
                          extra_compile_args=["-O3","-ffast-math"],
                          include_dirs=['figaro', lal_includes, numpy.get_include()]
                          ))
    else:
        ext_modules.append(Extension("figaro.cosmology",
                           sources=[os.path.join("figaro","cosmology.pyx")],
                           libraries=["m", "lal"], # Unix-like specific
                           extra_compile_args=["-O3","-ffast-math"],
                           include_dirs=['figaro', numpy.get_include()]
                           ))

ext_modules = cythonize(ext_modules, compiler_directives={'language_level' : "3"})

scripts = ['figaro-density=figaro.pipelines.probability_density:main',
           'figaro-hierarchical=figaro.pipelines.hierarchical_inference:main',
           'figaro-glade=figaro.pipelines.create_glade:main',
           'figaro-pp_plot=figaro.pipelines.ppplot:main',
           'figaro-mockdata=figaro.pipelines.gen_mock_data:main',
           'figaro-entropy=figaro.pipelines.entropy:main',
           ]
pymodules = ['figaro/pipelines/probability_density',
             'figaro/pipelines/hierarchical_inference',
             'figaro/pipelines/create_glade',
             'figaro/pipelines/ppplot',
             'figaro/pipelines/gen_mock_data',
             'figaro/pipelines/entropy',
             ]

par_scripts = ['figaro-par-hierarchical=figaro.pipelines.par_hierarchical_inference:main',
               'figaro-par-density=figaro.pipelines.par_probability_density:main',
               ]
par_modules = ['figaro/pipelines/par_hierarchical_inference',
               'figaro/pipelines/par_probability_density',
              ]

if ray_flag:
    scripts   = scripts + par_scripts
    pymodules = pymodules + par_modules

setup(
    name = 'figaro',
    description = 'FIGARO: Fast Inference for GW Astronomy, Research & Observations',
    author = 'Stefano Rinaldi, Walter Del Pozzo, Daniele Sanfratello',
    author_email = 'stefano.rinaldi@phd.unipi.it, walter.delpozzo@unipi.it, d.sanfratello@studenti.unipi.it',
    url = 'https://github.com/sterinaldi/figaro',
    python_requires = '>=3.7',
    packages = ['figaro'],
    py_modules = pymodules,
    install_requires=requirements,
    include_dirs = ['figaro', numpy.get_include()],
    setup_requires=['numpy', 'cython'],
    package_data={"": ['*.c', '*.pyx', '*.pxd']},
    ext_modules=ext_modules,
    entry_points = {
        'console_scripts': scripts,
        },
    version='1.0.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    )

setup(
      name = 'figaro/cumulative',
      ext_modules = cythonize(ext_modules, language_level = "3"),
      include_dirs=['figaro', numpy.get_include()]
      )
if lal_flag:
    setup(
          name = 'figaro/cosmology',
          ext_modules = cythonize(ext_modules, language_level = "3"),
          include_dirs=['figaro', numpy.get_include()]
          )


if not lal_flag:
    warnings.warn("\n\nWARNING: No LAL installation found, please install LAL - see https://wiki.ligo.org/Computing/LALSuiteInstall. Some functions - GW posterior samples loading and catalog loading - won't be available and errors might be raised.\n", stacklevel = 2)
if not ray_flag:
    warnings.warn("\n\nWARNING: Ray is not installed: parallelized pipelines won't be available. If you want to use them, please install Ray (pip install ray) and reinstall FIGARO.\n", stacklevel = 2)
