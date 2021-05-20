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
import codecs
import datetime
import os
import re
import sys

# sys.path.insert(0, os.path.abspath('.'))
sys.path.append(os.path.abspath("./_ext"))
sys.path.append(os.path.abspath(".."))


# -- Project information -----------------------------------------------------

def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, *parts), "rb", "utf-8") as f:
        return f.read()


def find_version(*file_paths):
    """
    Build a path from *file_paths* and search for a ``__version__``
    string inside.
    """
    version_file = read(*file_paths)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


project = "Eradiate"
copyright = f"2020-{datetime.datetime.now().year}, The Eradiate Team"
author = "The Eradiate Team"
release = find_version("../eradiate/__init__.py")
version = release.rsplit(u".", 1)[0]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = []

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

# Add custom extension
extensions.append("ertdocs")

rst_prolog = r"""
.. role:: bolditalic
  :class: bolditalic
"""

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_logo = "fig/icon_eradiate.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Configure extensions
extensions.append("sphinx.ext.mathjax")
extensions.append("sphinx.ext.viewcode")
extensions.append("sphinx_copybutton")

# Bibliography
extensions.append("sphinxcontrib.bibtex")
bibtex_bibfiles = ["references.bib"]

# Example gallery support
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

extensions.append("sphinx_gallery.gen_gallery")
sphinx_gallery_conf = {
    "examples_dirs": [  # paths to example scripts
        "examples/tutorials",
    ],
    "gallery_dirs": [  # path to where to save gallery generated output
        "examples/generated/tutorials/",
    ],
    "subsection_order": ExplicitOrder([
        "examples/tutorials/solver_onedim",
        "examples/tutorials/solver_rami",
        "examples/tutorials/atmosphere",
        "examples/tutorials/biosphere",
        "examples/tutorials/data",
    ]),
    "within_subsection_order": FileNameSortKey,
    "filename_pattern": "/",
    "reference_url": {
        "eradiate": None,  # The module you locally document uses None
    },
    "plot_gallery": False,  # Disabled until we move away from RTD or package the kernel
    "default_thumb_file": "fig/icon_eradiate.png",
}

# Sphinx-panels extension: don't load Bootstrap CSS again
extensions.append("sphinx_panels")
panels_add_bootstrap_css = False

# Intersphinx configuration for cross-project referencing
extensions.append("sphinx.ext.intersphinx")
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "xarray": ("http://xarray.pydata.org/en/stable/", None),
    "mitsuba": ("https://eradiate-kernel.readthedocs.io/en/latest/", None),
    "pint": ("https://pint.readthedocs.io/en/latest/", None),
    "attr": ("https://www.attrs.org/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pinttrs": ("https://pinttrs.readthedocs.io/en/latest/", None),
}

# Activate todo notes
extensions.append("sphinx.ext.todo")
todo_include_todos = True

# Autodoc and autosummary options
extensions.append("sphinx.ext.autodoc")
autodoc_default_flags = [
    "members", "undoc-members", "show-inheritance", "inherited-members"
]
autodoc_typehints = "none"

extensions.append("sphinx.ext.autosummary")
autosummary_generate = True
autosummary_members = True

# Mitsuba modules must be mocked in order to allow compiling docs even if they're not here;
# this mocking is also done in the ertdocs extension
autodoc_mock_imports = [
    "mitsuba", "mitsuba.core", "mitsuba.core.math", "mitsuba.core.spline",
    "mitsuba.core.warp", "mitsuba.core.xml", "mitsuba.render",
    "mitsuba.render.mueller"
]

# ------------------------- HTML output customisation -------------------------

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "github_url": "https://github.com/eradiate/eradiate"
}

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = []

# Register extra CSS
html_css_files = ["theme_overrides.css"]

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "Eradiate"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_sidebars = {}

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
# html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = '%b %d, %Y'

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Language to be used for generating the HTML full-text search index.
# Sphinx supports the following languages:
#   'da', 'de', 'en', 'es', 'fi', 'fr', 'h', 'it', 'ja'
#   'nl', 'no', 'pt', 'ro', 'r', 'sv', 'tr'
# html_search_language = 'en'

# A dictionary with options for the search language support, empty by default.
# Now only 'ja' uses this config value
# html_search_options = {'type': 'default'}

# The name of a javascript file (relative to the configuration directory) that
# implements a search results scorer. If empty, the default will be used.
# html_search_scorer = 'scorer.js'

# Output file base name for HTML help builder.
htmlhelp_basename = "eradiate_doc"

# ------------------------- LaTeX output customisation -------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    "papersize": "a4paper",

    "classoptions": ",oneside",

    # The font size ("10pt", "11pt" or "12pt").
    "pointsize": "10pt",

    # Fonts
    "fontpkg": r"""
        \setmainfont{Charis SIL}[Scale=.98]
        \setsansfont{Source Sans Pro}[Scale=MatchLowercase]
        \setmonofont{Hack}[Scale=MatchLowercase]
    """,

    # Additional stuff for the LaTeX preamble.
    "preamble": "",

    # Latex figure (float) alignment
    # 'figure_align': 'htbp',
}

latex_engine = "xelatex"

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    ("index_latex", "eradiate.tex", "Eradiate Documentation",
     "The Eradiate Team", "manual"),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = "fig/eradiate-logo-dark-no_bg.png"

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True
