# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import sphinx_rtd_theme
import os
import sys
from unittest import mock
sys.path.insert(0, os.path.abspath('../../sadaco/'))
# -- Project information -----------------------------------------------------

project = 'SADACO'
copyright = '2022, Kyungchae Lee, Ying Hui Tan'
author = 'Kyungchae Lee, Ying Hui Tan'

# The full version, including alpha/beta/rc tags
release = '0.1'

to_import = list(sys.modules.keys())
import torch
import torchaudio
import torchvision
# try:
#     import torch  # noqa
# except ImportError:
#     for m in [
#         "torch", "torchvision", "torch.nn", "torch.nn.parallel", "torch.distributed", "torch.multiprocessing", "torch.autograd",
#         "torch.autograd.function", "torch.nn.modules", "torch.nn.modules.utils", "torch.utils", "torch.utils.data", "torch.onnx",
#         "torch.cuda", "torch.utils.data.sampler", "torch.cuda.amp",'torch.nn', 'torch.nn.functional', 'torchsummary', 'torch.optim', 
#         'torch.optim.lr_scheduler', "torchvision", "torchvision.ops", 'torchvision.transforms', 'torchvision.transforms.functional', 
#     ]:
#         sys.modules[m] = mock.Mock(name=m)
#     sys.modules['torch'].__version__ = "1.8"  # fake version
#     HAS_TORCH = False

# try:
#     import torchaudio
# except ImportError:
#     for m in [
#         'torchaudio'
#     ]:
#         sys.modules[m] = mock.Mock(name=m)
#     sys.modules['torchaudio'].__version__ = "1.7"  # fake version
#     HAS_TORCHAUDIO = False

# current = list(sys.modules.keys())
# "tqdm", 'timm', 'timm.models', 'timm.models.layers', 'sklearn', 'sklearn.metrics','cProfile',
#      'matplotlib', 'matplotlib.pyplot', 'torch2trt', 'resampy', 'PIL', 'torchsummary'
for m in [
    "tqdm", 'timm', 'timm.models', 'timm.models.layers', 'torchsummary', 'torch2trt', 'soundfile'
]:
    sys.modules[m] = mock.Mock(name=m)

import sadaco
# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    'sphinx_rtd_theme','myst_parser'
]
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False
autodoc_inherit_docstrings = False
autodoc_member_order = "bysource"

source_suffix = [".rst", ".md"]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "build", "README.md", "tutorials/README.md"]

master_doc = "index"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
pygments_style = "sphinx"

html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']