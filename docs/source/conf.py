import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "DP5"
copyright = "2024, Kristaps Ermanis, Alexander Howarth, Ruslan Kotlyarov, Benji Rowlands, Jonathan Goodman"
author = "Kristaps Ermanis, Alexander Howarth, Ruslan Kotlyarov, Benji Rowlands, Jonathan Goodman"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
exclude_patterns = []
autodoc_mock_imports = [
    "numpy",
    "sklearn",
    "pandas",
    "tensorflow",
    "pathos",
    "scipy",
    "nmrglue",
    "matplotlib",
    "tomli",
    "lmfit",
    "openbabel",
    "statsmodels",
    "rdkit",
    "tqdm",
    "keras",
]
# autodoc_typehints = "description"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
