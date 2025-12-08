# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the parent directory to the path so we can import WindGym
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'WindGym'
copyright = '2024, Technical University of Denmark (DTU)'
author = 'Marcus Nilsen, Julian Quick, Ernestas Simutis, Teodor Olof Benedict Åstrand'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'myst_parser',
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_typehints = 'description'
autodoc_class_signature = 'separated'

# Autosummary settings
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'
html_static_path = []

# -- Options for Markdown output ---------------------------------------------
# Using myst_parser for markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Mock imports for dependencies that may not be installed
autodoc_mock_imports = [
    'gymnasium',
    'gym',
    'numpy',
    'np',
    'matplotlib',
    'plt',
    'dynamiks',
    'py_wake',
    'stable_baselines3',
    'hipersim',
    'pettingzoo',
    'wetb',
    'IPython',
    'h2lib',
    'xarray',
    'pandas',
    'yaml',
    'scipy',
    'torch',
    'tensorflow',
    'tqdm',
    'wandb',
    'tensorboard',
]
