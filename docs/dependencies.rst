.. _sec-dependencies:

Dependencies
============

Required dependencies
---------------------

* Python (3.9 to 3.13)

Core libraries

* `Mitsuba 3 <https://mitsuba.readthedocs.io/>`__ (radiometric engine; custom build,
  packaged as ``eradiate-mitsuba`` on PyPI)
* `Joseki <https://github.com/rayference/joseki>`__ (atmospheric profile
  management)
* `AxsDB <https://github.com/eradiate/axsdb/>`__ (molecular absorption databases)

Numerical computing infrastructure

* `NumPy <https://numpy.org/>`__
* `xarray <https://docs.xarray.dev>`__
* `SciPy <https://scipy.org/>`__
* `NetworkX <https://networkx.org/>`__ (graph-based pipeline management)

Visualization

* `Matplotlib <https://matplotlib.org/>`__

Unit handling

* `Pint <https://pint.readthedocs.io/>`__
* `Pinttrs <https://pinttrs.readthedocs.io/>`__

I/O and data management

* `netCDF4 <https://github.com/Unidata/netcdf4-python>`__
* `Pooch <https://www.fatiando.org/pooch/>`__
* `ruamel.yaml <https://yaml.readthedocs.io/>`__
* `cachetools <https://github.com/tkem/cachetools/>`__
* `Cerberus <https://python-cerberus.org/>`__ (data validation)

Class engine

* `attrs <https://www.attrs.org/>`__
* `Dessine-moi <https://dessinemoi.readthedocs.io/>`__
* `lazy_loader <https://github.com/scientific-python/lazy_loader>`__

Configuration

* `Dynaconf <https://github.com/dynaconf/dynaconf/>`__
* `aenum <https://github.com/ethanfurman/aenum>`__

Interface

* `Rich <https://rich.readthedocs.io/>`__
* `tqdm <https://github.com/tqdm/tqdm/>`__
* `Typer <https://typer.tiangolo.com/>`__
* `Click <https://click.palletsprojects.com/>`__

Optional dependencies
---------------------

Recommended
^^^^^^^^^^^

* `JupyterLab <https://jupyter.org/>`__,
  `ipython <https://ipython.org/>`__,
  `ipywidgets <https://ipywidgets.readthedocs.io/>`__: Highly recommended for
  interactive usage.
* `Seaborn <https://seaborn.pydata.org/>`__: Used to define the Eradiate plotting
  style.
* `pydot <https://github.com/pydot/pydot>`__: Used to visualize post-processing
  pipeline DAGs.
* `Skyfield <https://rhodesmill.org/skyfield/>`__,
  `python-dateutil <https://dateutil.readthedocs.io/>`__:
  Used for Earth-Sun distance calculation and date parsing in the Solar
  irradiance spectrum init code.
* `AABBTree <https://aabbtree.readthedocs.io/>`__: Used for collision detection
  in the leaf cloud generator.

Testing
^^^^^^^

* `pytest <https://docs.pytest.org/>`__
* `pytest-cov <https://pytest-cov.readthedocs.io/>`__
* `pytest-json-report <https://github.com/numirias/pytest-json-report>`__
* `pytest-robotframework <https://github.com/DetachHead/pytest-robotframework>`__
* `Robot Framework <https://robotframework.org/>`__
* `ASV <https://asv.readthedocs.io/>`__ (Airspeed Velocity; benchmarking)

Documentation
^^^^^^^^^^^^^

* `Sphinx <https://www.sphinx-doc.org/>`__
* `autodocsumm <https://autodocsumm.readthedocs.io/>`__ (vendored, see :ghpr:`410` for details)
* `myst-parser <https://myst-parser.readthedocs.io/>`__
* `nbsphinx <https://nbsphinx.readthedocs.io/>`__
* `shibuya <https://shibuya.lepture.com/>`__
* `sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io/>`__
* `sphinx-autobuild <https://github.com/executablebooks/sphinx-autobuild>`__
* `sphinx-copybutton <https://sphinx-copybutton.readthedocs.io/>`__
* `sphinx-design <https://sphinx-design.readthedocs.io/>`__
* `sphinx-iconify <https://github.com/lepture/sphinx-iconify>`__
