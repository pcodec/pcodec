import toml
# I don't know why, but sphinx fails to import pcodec during autodoc without this
import pcodec # noqa: F401

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# taking a lot of ideas from
# https://github.com/delta-io/delta-rs/blob/main/python/docs/source/conf.py

project = 'Pcodec'
copyright = '2025, Pcodec devs'
author = 'Pcodec devs'

def get_release_version() -> str:
    return toml.load("../Cargo.toml")["package"]["version"]

version = get_release_version()

# from pcodec import standalone, wrapped

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
