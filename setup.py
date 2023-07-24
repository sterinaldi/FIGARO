import numpy
from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from codecs import open
from pathlib import Path
from distutils.extension import Extension
import os
import warnings

ray_flag = True
try:
    import ray
except ModuleNotFoundError:
    ray_flag = False

try:
    from Cython.Build import cythonize, build_ext
except ImportError:
    raise ImportError("Cython not found. Please install it via\n\tpip install Cython")

if os.environ['CONDA_DEFAULT_ENV'] == 'igwn-py39':
    requirements = ['imageio']
else:
    with open("requirements.txt") as requires_file:
        requirements = requires_file.read().split("\n")

with open("README.md") as readme_file:
    long_description = readme_file.read()

ext_modules=[
             Extension("figaro.cumulative",
                       sources=[os.path.join("figaro","cumulative.pyx")],
                       libraries=["m"], # Unix-like specific
                       extra_compile_args=["-O3","-ffast-math"],
                       include_dirs=['figaro', numpy.get_include()]
                       ),
            ]

# VERY dirty solution to get LAL (but it works, so... Who cares?)
os.system('conda install -S --channel conda-forge lalsuite')
lal_folder = os.environ['CONDA_PREFIX']
ext_modules.append(Extension("figaro.cosmology",
                   sources=[os.path.join("figaro","cosmology.pyx")],
                   libraries=["m", "lal"], # Unix-like specific
                   extra_compile_args=["-O3","-ffast-math"],
                   library_dirs = [os.path.join(lal_folder, "lib/")],
                   include_dirs=['figaro', os.path.join(lal_folder, "include/"), numpy.get_include()]
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
    python_requires = '>=3.9',
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
    version='1.2.1',
    long_description=long_description,
    long_description_content_type='text/markdown',
    cmdclass = {
            "build_ext": build_ext
            }
    )

if not ray_flag:
    warnings.warn("\n\nWARNING: Ray is not installed: parallelized pipelines won't be available. If you want to use them, please install Ray (pip install ray) and reinstall FIGARO.\n", stacklevel = 2)
