.. _sec-developer_guide-dependencies:

Managing dependencies
=====================

Dependency management in a development environment requires care: loosely
specified dependencies allow for more freedom when setting up an environment,
but can also lead to reproducibility issues. To get a better understanding of
the underlying problems, the two following posts are interesting reads, which
the reader is strongly encouraged to study since most of the terminology used in
this guide comes from them:

* `Python Application Dependency Management in 2018 (Hynek Schlawak) <https://hynek.me/articles/python-app-deps-2018/>`_
* `Reproducible and upgradable Conda environments: dependency management with conda-lock (Itamar Turner-Trauring) <https://pythonspeed.com/articles/conda-dependency-management/>`_

Our dependency management system is designed with the following requirements:

1. Support for Conda: The system should be usable with Conda.
2. Support for Pip: The system should be usable with Pip.
3. Simplicity: The system must be usable by users with little knowledge of it.

Our system uses two tools:

* `conda-lock <https://github.com/conda-incubator/conda-lock>`_
* `pip-tools <https://github.com/jazzband/pip-tools>`_

Basic principles
----------------

We categorise our dependencies in four sets:

* ``main``: dependencies required to run Eradiate as a user;
* ``docs``: dependencies required to compile the documentation;
* ``tests``: dependencies required to run tests;
* ``dev``: dependencies specific to a development setup.

``main``, ``docs`` and ``tests`` are subsets of ``dev``:

- ``docs`` includes ``main``;
- ``tests`` includes ``main``;
- ``dev`` includes ``docs`` and ``tests``.

.. only:: latex

   .. figure:: ../../fig/requirement_layers.png

.. only:: not latex

   .. figure:: ../../fig/requirement_layers.svg

The sets are defined in ``setup.cfg``, where direct dependencies are specified
with minimal constraint.

.. warning:: This is the location from which all dependencies are sourced.
   Dependencies shoud all be specified only in ``setup.cfg``.

We then have processes which will compile these dependencies into transitively
pinned dependencies and write them as requirement (lock) files. The Conda and
Pip pinning processes are different.

The generate lock files are versioned and come along the source code they were
used to write. Thus, a developer cloning the codebase will also get the
information they need to reproduce the same environment as the other developers.

Lock files
----------

Lock files are stored in the ``requirements`` directory, alongside a series of
utility scripts.

* **Conda** dependencies are pinned using conda-lock. It uses a regular
  ``environment.yml`` file as input. It can compile requirements for multiple
  platforms, but cannot be used to extract subsets of an existing requirement
  specification. The ``environment.yml`` file is created by the
  ``make_conda_env.py`` script, from a header ``environment.in`` and the data
  found in ``setup.cfg``. Our Conda lock files use the extension ``.lock``.
* **Pip** dependencies are pinned using pip-tools. It uses a series of ``*.in``
  files as input (one per requirement set) which can be configured to define
  subsets of each other, but cannot compile requirements for multiple platforms,
  which basically means that we cannot use hashes to pin requirements with it.
  The ``*.in`` input files are created by the ``make_pip_in_files.py`` script
  from the data found in ``setup.cfg`` and the requirement layer relations
  defined in the ``layered.yml`` file. Our Pip lock files use the extension
  ``.txt``.

We can already see at this point that neither tool will perfectly fulfill our
requirements, but the limitations we have observed so far have not (yet)
proven to be critical.

Initialising or updating an environment
---------------------------------------

**With Conda**, use the following command in your active virtual environment:

.. code:: bash

   make conda-init

.. note:: This command also executes the ``copy_envvars.py`` script, which
   adds to your environment a script which will set environment variables
   upon activation.

**With Pip**, use the following command in your active virtual environment:

.. code:: bash

   make pip-init

These commands will use their respective package manager to update the currently
active environment with the pinned package versions.

Updating lock files
-------------------

When you want to update pinned dependencies (*e.g.* because you added or changed
a dependency in ``setup.cfg`` or because a dependency must be updated), you need
to update the lock file.

**With Conda**, use the following command in your active virtual environment:

.. code:: bash

   make conda-lock-all

**With Pip**, use the following command in your active virtual environment:

.. code:: bash

   make pip-lock

.. warning:: If you are developing in a Conda environment and want to update Pip
   lock files, use instead:

   .. code:: bash

      make pip-compile

   This command skips the Setuptools and pip-compile update which could disrupt
   your Conda environment.
