import datetime
import os
import sys
import unittest.mock as mock
from importlib.metadata import version

# -- Mock Mitsuba modules (required to render CLI reference on RTD) ------------

MOCK_MODULES = []
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

# -- Path setup ----------------------------------------------------------------

# sys.path.insert(0, os.path.abspath('.'))
sys.path.append(os.path.abspath("./_ext"))

# -- Project information -------------------------------------------------------

project = "Eradiate"
copyright = f"2020-{datetime.datetime.now().year}, The Eradiate Team"
author = "The Eradiate Team"
release = version("eradiate")
version = ".".join(release.split(".")[:3])
if "dev" in release:
    version += ".dev"

# -- General configuration -----------------------------------------------------


html_theme = "sphinx_book_theme"
html_title = ""
html_logo = "_static/eradiate-logo-typo-darkgrey.svg"

templates_path = ["_templates"]  # Path to templates, relative to this directory.
exclude_patterns = ["_build", "tutorials/README.md"]
html_static_path = ["_static"]

extensions = [
    # Core extensions
    "sphinx.ext.autodoc",  # Automatic API documentation
    "sphinx.ext.autosummary",  # Summary tables in API docs
    "sphinx.ext.doctest",  # Doctest blocks
    "sphinx.ext.extlinks",  # External links with dedicated roles
    "sphinx.ext.intersphinx",  # Inter-project links
    "sphinx.ext.mathjax",  # Equation support
    "sphinx.ext.napoleon",  # Better docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    # Third-party
    "autodocsumm",  # Possibly add autosummary table to autodoc
    "myst_parser",  # Markdown support
    "nbsphinx",  # Display notebooks
    "sphinx_copybutton",
    "sphinx_design",  # Tabs, buttons, icons...
    "sphinxcontrib.bibtex",  # BibTeX bibliography
    # Custom extensions
    "pluginparameters",  # Directives and roles to document Mitsuba plugins
]

# -- GitHub quicklinks with 'extlinks' -----------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/extlinks.html

ghbase = "https://github.com"
ghroot = f"{ghbase}/eradiate/eradiate"
extlinks = {
    "ghissue": (f"{ghroot}/issues/%s", "GH%s"),
    "ghpr": (f"{ghroot}/pull/%s", "PR%s"),
    "ghcommit": (f"{ghroot}/commit/%s", "%.7s"),
    "ghuser": (f"{ghbase}/%s", "@%s"),
}

# -- Bibliography with 'sphinxcontrib.bibtex' ----------------------------------
# https://sphinxcontrib-bibtex.readthedocs.io/en/latest/

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

# -- Tutorials -----------------------------------------------------------------
# https://nbsphinx.readthedocs.io/

if not os.path.exists("tutorials"):
    os.symlink("../tutorials", "tutorials", target_is_directory=True)
nbsphinx_execute = "never"
nbsphinx_prolog = """
.. button-link:: https://github.com/eradiate/eradiate-tutorials/blob/main/{{ env.doc2path(env.docname, base=False)|replace("tutorials/", "") }}
   :color: primary
   :expand:

   :fas:`external-link-alt` Go to notebook file

----
"""

# -- Intersphinx configuration for cross-project referencing -------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "attr": ("https://www.attrs.org/en/stable/", None),
    "attrs": ("https://www.attrs.org/en/stable/", None),
    "dateutil": ("https://dateutil.readthedocs.io/en/stable/", None),
    "dessinemoi": ("https://dessinemoi.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "mitsuba": ("https://mitsuba.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pooch": ("https://www.fatiando.org/pooch/latest/", None),
    "pint": ("https://pint.readthedocs.io/en/stable/", None),
    "pinttrs": ("https://pinttrs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "rich": ("https://rich.readthedocs.io/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

# -- Autodoc options -----------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

autodoc_default_flags = [
    "members",
    "undoc-members",
    "show-inheritance",
    "inherited-members",
]
autodoc_typehints = "none"
autodoc_member_order = "groupwise"
autodoc_preserve_defaults = False

# Mitsuba modules must be mocked in order to allow compiling docs even if they're not here;
# this mocking is also done in the ertdocs extension
autodoc_mock_imports = []

# Autosummary tables in autodoc
# https://autodocsumm.readthedocs.io/en/latest
autodoc_default_options = {
    # "autosummary": False,
}

# -- Autosummary options -------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html

autosummary_generate = True
autosummary_members = True

# -- Docstrings ----------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_custom_sections = [("Fields", "params_style")]
napoleon_preprocess_types = True
napoleon_type_aliases = {
    # general terms
    "callable": ":py:func:`callable`",
    "datetime": ":py:class:`~datetime.datetime`",
    "dict_like": ":term:`dict-like <mapping>`",
    "dict-like": ":term:`dict-like <mapping>`",
    "file-like": ":term:`file-like <file-like object>`",
    "iterable": ":term:`iterable`",
    "path-like": ":term:`path-like <path-like object>`",
    "mapping": ":term:`mapping`",
    "sequence": ":term:`sequence`",
    # stdlib type aliases
    "MutableMapping": "~collections.abc.MutableMapping",
    "Path": "~pathlib.Path",
    # numpy terms and aliases
    "array": ":term:`array`",
    "array_like": ":term:`array_like`",
    "array-like": ":term:`array-like <array_like>`",
    "dtype": "~numpy.dtype",
    "hashable": ":term:`hashable <name>`",
    "ndarray": "~numpy.ndarray",
    "scalar": ":term:`scalar`",
    "MaskedArray": "~numpy.ma.MaskedArray",
    # matplotlib terms and aliases
    "matplotlib axes": ":py:class:`matplotlib axes <matplotlib.axes.Axes>`",
    "axes": ":py:class:`axes <matplotlib.axes.Axes>`",
    "matplotlib figure": ":py:class:`matplotlib figure <matplotlib.figure.Figure>`",
    "figure": ":py:class:`figure <matplotlib.figure.Figure>`",
    # xarray terms and aliases
    "DataArray": "~xarray.DataArray",
    "Dataset": "~xarray.Dataset",
    "Variable": "~xarray.Variable",
    # pint terms and aliases
    "quantity": ":class:`quantity <pint.Quantity>`",
    # pandas terms and aliases
    "pd.Index": "~pandas.Index",
    "pd.NaT": "~pandas.NaT",
    # local aliases
    "AUTO": ":data:`~eradiate.attrs.AUTO`",
}

# -- Extra roles (requires the 'pluginparameters' extension) -------------------

rst_prolog = r"""
.. role:: bolditalic
   :class: font-weight-bold font-italic

.. role:: monospace
   :class: text-monospace

.. role:: monobold
   :class: text-monospace font-weight-bold

.. role:: monosp
   :class: text-monospace

.. role:: paramtype
   :class: text-monospace

.. |spectrum| replace:: :paramtype:`spectrum`
.. |texture| replace:: :paramtype:`texture`
.. |float| replace:: :paramtype:`float`
.. |bool| replace:: :paramtype:`boolean`
.. |int| replace:: :paramtype:`integer`
.. |false| replace:: :monosp:`false`
.. |true| replace:: :monosp:`true`
.. |string| replace:: :paramtype:`string`
.. |bsdf| replace:: :paramtype:`bsdf`
.. |phase| replace:: :paramtype:`phase`
.. |point| replace:: :paramtype:`point`
.. |vector| replace:: :paramtype:`vector`
.. |transform| replace:: :paramtype:`transform`
.. |volume| replace:: :paramtype:`volume`
.. |tensor| replace:: :paramtype:`tensor`

.. |exposed| replace:: :abbr:`P (This parameters will be exposed as a scene parameter)`
.. |differentiable| replace:: :abbr:`∂ (This parameter is differentiable)`
.. |discontinuous| replace:: :abbr:`D (This parameter might introduce discontinuities. Therefore it requires special handling during differentiation to prevent bias (e.g. prb-reparam)))`
"""

# -- HTML output customisation -------------------------------------------------

html_theme_options = {
    "repository_url": "https://github.com/eradiate/eradiate",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_issues_button": True,
    "logo": {
        "image_light": "_static/eradiate-logo-typo-black.svg",
        "image_dark": "_static/eradiate-logo-typo-white.svg",
    },
}

html_css_files = ["theme_overrides.css"]
html_short_title = "Eradiate"
html_favicon = "fig/icon_eradiate.png"
html_show_sourcelink = False
htmlhelp_basename = "eradiate_doc"

# -- LaTeX output customisation ------------------------------------------------

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
latex_logo = "fig/eradiate-logo-typo-black.png"

latex_documents = [
    (
        "index_latex",
        "eradiate.tex",
        "Eradiate Documentation",
        "The Eradiate Team",
        "manual",
    ),
]


# -------------------------- Custom generation steps ---------------------------


def custom_step(app):
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import generate_md_cli
    import generate_rst_api
    import generate_rst_plugins

    generate_md_cli.generate()  # CLI reference
    generate_rst_plugins.generate()  # Plugins
    generate_rst_api.generate_env_vars_docs()  # Environment variables
    generate_rst_api.generate_factory_docs()  # Factories


def setup(app):
    app.connect("builder-inited", custom_step)
