# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FIGARO'
copyright = '2023, Stefano Rinaldi'
author = 'Stefano Rinaldi'

# -- Path setup
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']

extensions = [
'myst_parser',
'sphinx.ext.autodoc',
'sphinx.ext.autosummary',
'sphinx.ext.viewcode',
'sphinx.ext.napoleon',
'nbsphinx',
'sphinx.ext.mathjax',
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
