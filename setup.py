import numpy
from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from codecs import open
from os import path
from distutils.extension import Extension
import os

try:
    from Cython.Build import cythonize
except ModuleNotFoundError:
    print('Cython not found. Please install it via\n\tpip install Cython')

# see https://stackoverflow.com/a/21621689/1862861 for why this is here
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        self.include_dirs.append(numpy.get_include())

ext_modules=[
             Extension("online_skyloc.coordinates",
                       sources=[os.path.join("online_skyloc","coordinates.pyx")],
                       libraries=["m"], # Unix-like specific
                       extra_compile_args=["-O3","-ffast-math"],
                       include_dirs=['online_skyloc', numpy.get_include()]
                       ),
#             Extension("online_skyloc.transform",
#                       sources=[os.path.join("online_skyloc","transform.pyx")],
#                       libraries=["m"], # Unix-like specific
#                       extra_compile_args=["-O3","-ffast-math"],
#                       include_dirs=['online_skyloc', numpy.get_include()]
#                       ),
             ]
ext_modules = cythonize(ext_modules, compiler_directives={'language_level' : "3"})

setup(
    name = 'hdpgmm',
    use_scm_version=True,
    description = 'Online sky localisation',
    author = 'Walter Del Pozzo, Stefano Rinaldi',
    author_email = 'walter.delpozzo@unipi.it, stefano.rinaldi@phd.unipi.it',
    url = 'https://git.ligo.org/stefano.rinaldi/online-localisation',
    python_requires = '>=3.7',
    packages = ['online_skyloc'],
    include_dirs = [numpy.get_include()],
    setup_requires=['numpy', 'cython', 'setuptools_scm'],
    entry_points={},
    package_data={"": ['*.c', '*.pyx', '*.pxd']},
    ext_modules=ext_modules,
    )
