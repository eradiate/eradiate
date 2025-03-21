.. _sec-maintainer_guide:

Maintainer guide
================

.. _sec-maintainer_guide-data:

Managing data
-------------

This section describes how the data shipped with Eradiate is managed.

Overview
^^^^^^^^

.. _eradiate-data: https://github.com/eradiate/eradiate-data/

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
    A directory of files versioned in the `eradiate-data`_ GitHub repository.
    This data store contains small files, and implementing reproducibility with
    it is fairly simple. This is the location where data goes by default, *i.e.*
    when it is small enough to fit there. The ``small_files`` data store is
    accessed offline in a development setup, *i.e.* when the Eradiate repository
    and its submodules are cloned locally, or remotely when Eradiate is
    installed is user mode. This data store holds a *registry* of files which it
    uses for integrity checks when it is accessed online. Only files in the
    registry can be served by this data store, regardless if it is accessed
    online or offline: files of the eradiate-data repository which are not
    registered cannot be accessed through it.

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

.. _pre-commit: https://pre-commit.com/

Each store requires a different protocol.

``small_files``
    Install `pre-commit`_ and install the git hook scripts:

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

.. _Pixi: https://pixi.sh/
.. _Pixi Basic usage: https://pixi.sh/latest/basic_usage/
.. _Pixi Python development: https://pixi.sh/latest/tutorials/python/

Eradiate is managed using the `Pixi`_ project manager. It notably allows
us to maintain a Conda-based setup with most dependencies sourced from PyPI,
with layered requirements. Be sure to read the relevant entries in the Pixi
documentation:

* `Pixi Basic usage`_
* `Pixi Python development`_

We use the following requirement groups, which manifest as `features` in the
Pixi model:

* ``optional``: Only contains the Eradiate kernel package. This is needed to
  allow developers to install all dependencies except the kernel.
* ``recommended``: Optional packages used by specific subcomponents of Eradiate.
* ``docs``: Packages needed to compile the documentation.
* ``test``: Packages needed to run the test suite.

When adding new requirements, be sure to:

* Prioritize PyPI packages, using the ``--pypi`` option.
* Register the new requirement to the appropriate group.
* Check in the lock file after it is updated.

.. note::

   Our Read The Docs build uses stock Python virtual environments and installs
   dependencies with Pip. A specific requirement file is generated for that
   purpose, using the ``docs-lock`` Pixi task. This task is automatically
   executed upon committing, so the file is always up-to-date.

.. _sec-maintainer_guide-release:

Making a release of Eradiate
----------------------------

.. _start a Pixi shell: https://pixi.sh/latest/features/environment/#activation
.. _Bump My Version: https://github.com/callowayproject/bump-my-version
.. [1] This applies only if the Pixi environment is not activated already, *e.g.*
       by a ``direnv`` script.

1. **Preparation**

   1. Make sure main is up-to-date and all tests pass.
   2. If necessary, [1]_ `start a Pixi shell`_:

      .. code:: shell

         pixi shell -e dev

   3. In your shell, set the variable ``RELEASE_VERSION`` to the target version
      value:

      .. code:: shell

         export RELEASE_VERSION=X.Y.Z

   4. Create a new branch for the release:

      .. code:: shell

         git checkout main && git pull upstream main && git checkout -b bump/prepare-v$RELEASE_VERSION

   5. Make sure that dependencies are correct (check in particular the kernel
      version). Use the release checker utility for this:

      .. code:: shell

         python resources/release.py check-mitsuba

   6. Bump the version number using `Bump My Version`_:

      .. code:: shell

         pixi run bump

   7. Update the change log.
   8. Commit the changes:

      .. code:: shell

         git commit -am "Bump version to $RELEASE_VERSION"

   9. Update the version and release date fields in ``CITATION.cff``:

      .. code:: shell

         python resources/release.py update-citation

   10. Update the PyPI README content:

       .. code:: shell

          python resources/release.py update-pypi-readme

   11. Push the changes:

       .. code:: shell

          git push origin

2. **Pull request**

   1. Create a pull request to check changes with peers.
   2. Merge the pull request once everything is correct.

3. **Release publication**

   1. Create a draft release on GitHub and update it.
   2. Using release candidates on Test PyPI, make sure that built Pyhon wheels
      will work as expected. A typical installation command of a Test PyPI
      release is

      .. code:: shell

         python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ 'eradiate[kernel]==0.30.0rc4'

   3. Finalize release notes and create the release tag. **Make sure that the
      release commit is referenced only by one tag.**
   4. Build and upload Python wheels.

4. **Post-release: Prepare the next development cycle**

   1. In your shell, set the variable ``RELEASE_VERSION`` to the target version
      value:

      .. code:: shell

         export RELEASE_VERSION=X.Y.Z-dev0

   2. Bump the version number using:

      .. code:: shell

         pixi run bump
