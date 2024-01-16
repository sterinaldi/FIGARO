import numpy as np
import os
from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules=[
Extension("figaro.cumulative",
          sources=["figaro/cumulative.pyx"],
          libraries=["m"], # Unix-like specific
          extra_compile_args=["-O3","-ffast-math"],
          include_dirs=['figaro', np.get_include()],
          ),
Extension("figaro.cosmology",
          sources=["figaro/cosmology.pyx"],
          libraries=["m", "lal"], # Unix-like specific
          extra_compile_args=["-O3","-ffast-math"],
          library_dirs = [os.path.join(os.environ['CONDA_PREFIX'], "lib/")],
          include_dirs=['figaro', os.path.join(os.environ['CONDA_PREFIX'], "include/"), np.get_include()],
        ),
]
setup(ext_modules=cythonize(ext_modules, compiler_directives={'language_level' : "3"}))
