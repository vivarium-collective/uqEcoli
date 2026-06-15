# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the parent directory to the path so Sphinx can find the uq package
sys.path.insert(0, os.path.abspath("../.."))
# Add the repo root so autodoc can import the (real) ``inference`` package.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "vEcoli UQ Framework"
copyright = "2026, vEcoli Team"
author = "Alex Patrie"
release = "0.3.0"
version = "0.3.0"

# -- Mock imports for ReadTheDocs builds ------------------------------------
# RTD doesn't have vEcoli, PyTUQ, or other heavy dependencies installed.
# Mock them so autodoc can still parse type annotations and docstrings.
autodoc_mock_imports = [
    "ecoli",
    "pytuq",
    "pydmd",
    "polars",
    "duckdb",
    "scipy",
    "plotly",
    "textual",
    "typer",
    "click",
    "rich",
    "bigraph_schema",
    "process_bigraph",
    "vivarium",
    "ray",
    "salib",
    "marimo",
    "libuq",
    "uq",
    "v2ecoli",
    # Inference (Layer B/C) heavy deps — imported lazily inside methods, mocked
    # here so autodoc can parse the ``inference`` package without importing them.
    "ezyrb",
    "sbi",
    "torch",
    "emcee",
]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "polars": ("https://pola-rs.github.io/polars/py-polars/html/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

html_theme_options = {
    "dark_css_variables": {
        "color-brand-primary": "#e91e90",
        "color-brand-content": "#00e5ff",
    },
    "light_css_variables": {
        "color-brand-primary": "#b3006b",
        "color-brand-content": "#0088aa",
    },
    "navigation_with_keys": True,
    # Force dark mode as default (users can still toggle via sun/moon button)
    "default_mode": "dark",
}

# Pygments syntax highlighting: light for light mode, monokai for dark
pygments_style = "sphinx"
pygments_dark_style = "monokai"

# Custom CSS
html_css_files = [
    "custom.css",
]

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
}
