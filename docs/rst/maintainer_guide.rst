.. _sec-maintainer_guide:

Maintainer guide
================

.. _sec-maintainer_guide-data:

Managing data
-------------

This section describes how the data shipped with Eradiate is managed.

Overview
^^^^^^^^

Eradiate ships data of various sizes and maturity levels. The current data
management system tries to achieve a compromise between ease of use,
reproducibility and bandwidth and storage efficiency.

The entry point to Eradiate's data management system is the :mod:`eradiate.data`
module. The :mod:`~eradiate.data.open_dataset` and
:mod:`~eradiate.data.load_dataset` functions serve data retrieved from
:term:`data stores <data store>`. Eradiate currently has three data stores
aggregated as a :class:`.MultiDataStore` instance, accessible as the
:data:`eradiate.data.data_store` member. Each one of these aggregated data
stores is referenced with an identifier:

``small_files``
    A directory of files versioned in the
    `eradiate-data <https://github.com/eradiate/eradiate-data>`_ GitHub
    repository. This data store contains small files, and implementing
    reproducibility with it is fairly simple. This is the location where data
    goes by default, *i.e.* when it is small enough to fit there. The
    ``small_files`` data store is accessed offline in a development setup,
    *i.e.* when the Eradiate repository and its submodules are cloned locally,
    or remotely when Eradiate is installed is user mode. This data store holds
    a *registry* of files which it uses for integrity checks when it is accessed
    online. Only files in the registry can be served by this data store,
    regardless if it is accessed online or offline: files of the eradiate-data
    repository which are not registered cannot be accessed through it.

``large_files_stable``
    A directory of files hosted remotely. The files in this data store are
    expected to be too large to be conveniently stored in a Git repository. The
    data store holds a registry, used for integrity checks upon download, and it
    contains stable files. This means that the files should not be modified: if
    data is to be changed, it should be saved as a new file. This data store
    is expected to guarantee reproducibility. Unregistered files can also not
    be served by this data store.

``large_files_unstable``
    Another directory of files hosted remotely. Like for ``large_files_stable``,
    the files in this data store are expected to be too large to be conveniently
    stored in a Git repository. Files there are not registered: any query
    leading to data being downloaded will be considered as successful. This
    store does *not* guarantee reproducibility! In particular, this is the
    location where experimental data sets are located. Files stored there should
    not be used to write tests.

The :data:`~eradiate.data.data_store` is queried by passing paths to the desired
resources (relative to the root of the registered data stores) to the
:meth:`data_store.fetch() <eradiate.data.MultiDataStore.fetch>` method.
The aggregated stores are successively queried, and the outcome of the first
successful query is returned.

All online stores implement the following features which help reduce the amount
of online storage and traffic:

* caching: requested data is automatically downloaded and cached locally;
* lazy download: if data is already available locally, it is not downloaded
  again;
* compressed data substitution: upon query, online stores first check if a file
  with the same name and the ``.gz`` extension is available; if so, that file is
  downloaded, then automatically decompressed locally and served.

File registries, as mentioned earlier, are used for integrity checks when
downloading. They are also used to check if data has changed: if the online hash
value of the requested resource is different from the hash of the file in the
local cache directory, the file is downloaded again.

.. note::
   The ``large_files_unstable`` data store has no hash check: this means that
   refreshing its local cache can only be achieved by deleting its contents.

Modifying the data
^^^^^^^^^^^^^^^^^^

Each store requires a different protocol.

``small_files``
    Install `pre-commit <https://pre-commit.com/>`_ and install the git hook
    scripts:

    .. code:: bash

       cd $ERADIATE_SOURCE_DIR/resources/data
       pre-commit install

    Now add some data and commit your changes:

    .. code:: bash

       git checkout -b my_branch
       git add some_data.nc
       git commit -m "Added some data"

    The output should look something like:

    .. code:: bash

       Update registry..........................................................Failed
       - hook id: update-registry
       - files were modified by this hook

       Creating registry file from '.'
       Using rules in 'registry_rules.yml'
       Writing registry file to 'registry.txt'
       100% 181/181 [00:00<00:00, 100859.44it/s]

    The hook script failed because we changed the data and the changes were not commited.
    This is the expected behaviour.
    The hook script updated the registry file with the sha256 sum of the data file we added.
    Now add the changes to the registry file and commit again:

    .. code:: bash

       git add registry.txt
       git commit -m "Added some data"

    This time, the output should look something like:

    .. code::

       Update registry..........................................................Passed
       [master 0b9c760] Added some data
       2 files changed, 2 insertions(+)
       create mode 100644 spectra/some_data.nc

    The rules used to create the registry file are defined in the
    ``"registry_rules.yml"`` file.
    Be aware that if you add a data file that is not included by these rules, it will
    not be registered and therefore it will not be accessible by the data store.

    If, for some reason, you cannot use pre-commit, then you must be very careful and
    update the registry manually using the ``eradiate data make-registry``
    command-line tool (it should be run in the data submodule).

``large_files_stable``
    The most complicated: avoid updating the files, just add new ones. When
    doing so, you have to update the registry: compute the sha256 hash of the
    new file (*e.g.* ``sha256sum`` command-line tool) and update the registry
    file with this new entry. If you happen to have the full contents of the
    data store on your hard drive, you may also use the
    ``eradiate data make-registry`` command-line tool to update the registry
    automatically.

``large_files_unstable``
    The simplest: just drop the file in the remote storage, it will be
    immediately accessible.

.. _sec-maintainer_guide-dependencies:

Managing dependencies
---------------------

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

Our system uses two tools (included in the development virtual environment):

* `conda-lock <https://github.com/conda-incubator/conda-lock>`_
* `pip-tools <https://github.com/jazzband/pip-tools>`_

Basic principles
^^^^^^^^^^^^^^^^

We categorize our dependencies in five layers:

* ``main``: minimal requirements for eradiate to run in development mode
* ``recommended``: convenient optional dependencies included in the production package. Installable through PyPI.
* ``docs``: dependencies required to compile the docs in development mode
* ``tests``: dependencies required for testing eradiate in development mode
* ``dev``: dependencies specific to a development setup.
* ``dependencies``: dependency list used by default by Setuptools in production packages. Includes the ``eradiate-mitsuba`` package. Used by users who install Eradiate through PyPI.
* ``optional``: convenience development dependencies, including the ``eradiate-mitsuba`` package.

Layers can include other layers. As a result, we have the following layer Directed Acyclic Graph (DAG):

- ``docs`` includes ``main``;
- ``tests`` includes ``main``;
- ``dev`` includes ``recommended``, ``docs`` and ``tests``.
- ``dependencies`` includes ``main``;
- ``optional`` includes ``dev``;

The following figure illustrates the layer DAG:

.. only:: latex

   .. figure:: ../fig/requirement_layers.png

.. only:: not latex

   .. figure:: ../fig/requirement_layers.svg

The sets are defined in ``requirements/layered.yml``, where direct dependencies are
specified with minimal constraint.

.. warning:: This is the location from which all dependencies are sourced.
   Dependencies shoud all be specified only in ``requirements/layered.yml``.

We then have processes which will compile these dependencies into transitively
pinned dependencies and write them as requirement (lock) files. The Conda and
Pip pinning processes are different.

The generated lock files are versioned and come along the source code they were
used to write. Thus, a developer cloning the codebase will also get the
information they need to reproduce the same environment as the other developers.

The project's ``pyproject.toml`` file defines the metadata used by the Eradiate wheels.
It thus includes the necessary pip lock files for production/users setups. These are
the ``dependencies`` layer pip lock file, which includes the eradiate-mitsuba package, and
the ``recommended`` layer pip lock file, as an optional dependency set.

Lock files
^^^^^^^^^^

Lock files are stored in the ``requirements`` directory, alongside a series of
utility scripts.

* **Conda** dependencies are pinned using conda-lock. It uses a regular
  environment YAML file as input. It can compile requirements for multiple
  platforms, but cannot be used to extract subsets of an existing requirement
  specification. The ``environment-dev.yml`` file is created by the
  ``make_conda_env.py`` script, from a header ``environment.in`` and the data
  found in ``requirements/layered.yml``. Our Conda lock files use the extension ``.lock``.
* **Pip** dependencies are pinned using pip-tools. It uses a series of ``*.in``
  files as input (one per requirement layer) which can be configured to define
  subsets of each other, but cannot compile requirements for multiple platforms,
  which basically means that we cannot use hashes to pin requirements with it.
  The ``*.in`` input files are created by the ``make_pip_in_files.py`` script
  from the data found in ``requirements/layered.yml`` and the requirement layer relations
  defined in the ``requirements/layered.yml`` file. Our Pip lock files use the extension
  ``.txt``.

We can already see at this point that neither tool will perfectly fulfill our
requirements, but the limitations we have observed so far have not (yet)
proven to be critical.

Initialising or updating an environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^

When you want to update pinned dependencies (*e.g.* because you added or changed
a dependency in ``pyproject.toml`` or because a dependency must be updated), you
need to update the lock file.

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

Continuous integration
----------------------

Eradiate has a continuous integration scheme built in `Github Actions <https://docs.github.com/en/actions>`_ .
The action is configured in the ``.github/workflows/ci.yml`` file.

As per the documented installation process, Conda environment setup is handled using
the appropriate Makefile and Mitsuba build configuration is done using the CMake preset.
No CI-specific build setup operations are required.

The CI workflow uses caching for the compiled Mitsuba binaries. The cache is identified by the commit hash of the 
``mitsuba`` submodule and the file hashes of all .cpp and .h files in ``src/plugins/src``.

Since the entire pipeline takes more than one hour to complete, it is not triggered automatically.
Instead, issuing a PR comment containing only ``run Eradiate CI`` will trigger the pipeline on the source
branch of the PR.

.. _sec-maintainer_guide-release:

Preparing a  release
--------------------

1. Update the change log.
2. Tag the target commit for release.

Tagging a commit for release
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Eradiate picks up its version number using the `setuptools-scm <https://github.com/pypa/setuptools_scm>`_
package. Under the hood, it uses Git tags and the ``git describe`` command,
which only picks up annotated tags. To make sure that the tags will be
correctly picked up,
`make sure that they are annotated <https://stackoverflow.com/questions/4154485/git-describe-ignores-a-tag>`_
using

.. code:: bash

   git tag -a <your_tag_name> -m "<your_message>"

Note that the message may be an empty string.
