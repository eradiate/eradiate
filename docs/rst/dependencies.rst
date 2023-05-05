.. _sec-dependencies:

Dependencies
============

Required dependencies
---------------------

* Python (3.8 or later)

Radiometric engine

* `Mitsuba 3 <https://mitsuba.readthedocs.io/>`_

Numerical computing infrastructure

* `NumPy <https://numpy.org/>`_
* `xarray <https://docs.xarray.dev>`_
* `SciPy <https://scipy.org/>`_

Unit handling

* `Pint <https://pint.readthedocs.io/>`_
* `Pinttrs <https://pinttrs.readthedocs.io/>`_

I/O and data management

* `netCDF4 <https://github.com/Unidata/netcdf4-python>`_
* `Pooch <https://www.fatiando.org/pooch/>`_
* `ruamel.yaml <https://yaml.readthedocs.io/>`_

Class engine

* `attrs <https://www.attrs.org/>`_
* `Dessine-moi <https://dessinemoi.readthedocs.io/>`_

Configuration

* `environ-config <https://environ-config.readthedocs.io/>`_
* `aenum <https://github.com/ethanfurman/aenum>`_

Interface

* `Rich <https://rich.readthedocs.io/>`_
* `tqdm <https://github.com/tqdm/tqdm/>`_
* `Typer <https://typer.tiangolo.com/>`_

Optional dependencies
---------------------

Recommended
^^^^^^^^^^^

* `JupyterLab <https://jupyter.org/>`_,
  `ipython <https://ipython.org/>`_,
  `ipywidgets <https://ipywidgets.readthedocs.io/>`_: Highly recommended for
  interactive usage.
* `Matplotlib <https://matplotlib.org/>`_: Highly recommended, default plotting
  library for xarray.
* `Seaborn <https://seaborn.pydata.org/>`_: Used to define the Eradiate plotting
  style.

Testing
^^^^^^^

* `pytest <https://docs.pytest.org/>`_
* `pytest-json-report <https://github.com/numirias/pytest-json-report>`_

Maintenance
^^^^^^^^^^^

* `conda-lock <https://github.com/conda-incubator/conda-lock>`_
* `pip-tools <https://pip-tools.readthedocs.io>`_

Documentation
^^^^^^^^^^^^^

* `Sphinx <https://www.sphinx-doc.org/>`_
* `autodocsumm <https://autodocsumm.readthedocs.io/>`_
* `myst-parser <https://myst-parser.readthedocs.io/>`_
* `nbsphinx <https://nbsphinx.readthedocs.io/>`_
* `sphinx-book-theme <https://sphinx-book-theme.readthedocs.io/>`_
* `sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io/>`_
* `sphinx-autobuild <https://github.com/executablebooks/sphinx-autobuild>`_
* `sphinx-copybutton <https://sphinx-copybutton.readthedocs.io/>`_
* `sphinx-design <https://sphinx-design.readthedocs.io/>`_

Others
^^^^^^

* `AABBTree <https://aabbtree.readthedocs.io/>`_: Used for collision detection
  in the lead cloud generator.
* `astropy <https://docs.astropy.org/>`_,
  `python-dateutil <https://dateutil.readthedocs.io/>`_:
  Used for Earth-Sun distance calculation and date parsing in the Solar
  irradiance spectrum init code.
* `IAPWS <https://iapws.readthedocs.io/>`_: Used for thermophysical property
  computations.
