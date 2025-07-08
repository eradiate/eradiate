.. _sec-data-intro:

Data guide introduction
=======================

Input data plays a role equally important to the radiative transfer equation
integration algorithm when it comes to simulating radiative transfer. Data
handling and procurement in Eradiate follows the following principles:

* Eradiate reads standard data formats that are understood by the main data
  processing libraries of the scientific Python ecosystem. Most data are
  supplied in the `NetCDF format <https://www.unidata.ucar.edu/software/netcdf/>`_
  and loaded as `xarray <https://xarray.dev/>`_ datasets. Xarray provides a
  comprehensive, robust and convenient interface to read, write, manipulate and
  visualize NetCDF data.

* Eradiate ships data to use as "sensible defaults", similar to most radiative
  transfer models. We try to make shipped data transparent and take advantage of
  modern documentation facilities to make it comprehensive.

* Eradiate lets users swap their input with data sourced by themselves. If you
  want to use your own surface reflection spectra, aerosol single-scattering
  properties or molecular absorption data, it is possible thanks to the
  documented data formats.

* Eradiate provides an interface to facilitate the delivery of shipped data to
  users. Downloading a dataset is usually not more complicated than issuing a
  single command line in a terminal.

Basic concepts
--------------

Data handling is split into two parts:

* Data consumption: this is the process of delivering data that are available
  locally to the components of Eradiate.

* Data shipping: this is the process of delivering to the user data delivered
  together with Eradiate.

Each phase of data handling is supported by a specific component, each available
as a unique global instance:

* The :class:`.FileResolver` resolves relative paths by searching an ordered
  list of registered local directories. It allows to maintain shipped data as a
  relocatable file tree.

* The :class:`.AssetManager` manages shipped data. It connects to an online data
  registry that publishes a list of available resources. It can download,
  decompress and install available resources to a configurable local directory
  that is appended to the file resolver automatically.
  It is accessed through the unique instance :data:`eradiate.asset_manager`, but
  the main interaction point from a user point of view is the ``eradiate data``
  command-line utility.

.. _sec-data-intro-download:

Downloading data
----------------

Data is managed with the ``eradiate data`` command-line utility
(see :ref:`sec-reference_cli`). Known resources are displayed using the
``eradiate data list`` command, *e.g.*:

.. code-block:: console

    $ eradiate data list

      Resource ID                   Type     Size      State
     ────────────────────────────────────────────────────────
      absorption_ckd/monotropa-v1   tar.gz   57.1 MB   ---
      absorption_ckd/mycena-v1      tar.gz   126 MB    ---
      absorption_ckd/mycena-v2      tar.gz   87.2 MB   ---
      absorption_ckd/panellus-v1    tar.gz   790 MB    ---
      absorption_mono/gecko-v1      tar.gz   311 MB    ---
      absorption_mono/komodo-v1     tar.gz   235 MB    ---
      aerosol/core-v1               tar.gz   2.02 MB   ---
      bsdf/core-v1                  tar.gz   35.8 kB   ---
      constant/core-v1              tar.gz   2.74 kB   ---
      solar_irradiance/core-v1      tar.gz   2.97 MB   ---
      solar_irradiance/solid-v1     tar.gz   39.2 MB   ---
      srf/core-v1                   tar.gz   2.75 MB   ---
      texture/core-v1               tar.gz   5.55 kB   ---

.. tip::

    Upon a call to ``eradiate data list``, the registry's *manifest file* is
    updated. This manifest contains the list of resources that are available
    from the remote data registry. By default, registry updates happen only if
    the registry is more than one day old. A forced update is however
    possible, using ``eradiate data update``.

To install a given resources, use the ``eradiate data install`` command,
referencing the target resources by their IDs, *e.g.*:

.. code-block:: console

    $ eradiate data install aerosol/core-v1 bsdf/core-v1
    Downloading data from 'https://eradiate-data-registry.s3.eu-west-3.amazonaws.com/registry-v1/aerosol/core-v1.tar.gz' to file '/home/leroyv/.cache/eradiate/cached/aerosol/core-v1.tar.gz'.
    100%|█████████████████████████████████████| 2.02M/2.02M [00:00<00:00, 12.5GB/s]
    Untarring contents of '/home/leroyv/.cache/eradiate/cached/aerosol/core-v1.tar.gz' to '/home/leroyv/.cache/eradiate/unpacked/aerosol'
    Downloading data from 'https://eradiate-data-registry.s3.eu-west-3.amazonaws.com/registry-v1/bsdf/core-v1.tar.gz' to file '/home/leroyv/.cache/eradiate/cached/bsdf/core-v1.tar.gz'.
    100%|██████████████████████████████████████| 35.8k/35.8k [00:00<00:00, 265MB/s]
    Untarring contents of '/home/leroyv/.cache/eradiate/cached/bsdf/core-v1.tar.gz' to '/home/leroyv/.cache/eradiate/unpacked/bsdf'
    Installing resource 'aerosol/core-v1'
    Installing resource 'bsdf/core-v1'

Resource archives that are not already available locally will be downloaded from
the remote data registry. They will be unpacked and linked to the version-
dependent installation directory. The cache and unpacking locations are, by
default, common to all Eradiate versions. That means that if you use default
settings, data that was downloaded with a given version of Eradiate will not be
downloaded or unpacked again after an upgrade — only the symbolic links created
in the version-specific installation directory will be created again.

For convenience, some resources are aliased. The list of aliases can be
displayed as follows:

.. code-block:: console

    $ eradiate data list --what aliases

      Alias              Target
     ────────────────────────────────────────────────
      aerosol            aerosol/core-v1
      bsdf               bsdf/core-v1
      constant           constant/core-v1
      gecko              absorption_mono/gecko-v1
      solar_irradiance   solar_irradiance/core-v1
      komodo             absorption_mono/komodo-v1
      monotropa          absorption_ckd/monotropa-v1
      mycena             absorption_ckd/mycena-v2
      panellus           absorption_ckd/panellus-v1
      srf                srf/core-v1
      texture            texture/core-v1
      core               aerosol
                         bsdf
                         constant
                         komodo
                         monotropa
                         solar_irradiance
                         srf
                         texture
      absorption         gecko
                         komodo
                         monotropa
                         panellus
                         mycena

Some aliases reference a single resource, while others reference multiple
resources or aliases. It is usually recommended to download the ``core``
resources after installation.

The ``eradiate data`` command shows configuration information for the assert
manager and the file resolver:

.. code-block:: console

    $ eradiate data

    ── Asset manager ─────────────────────────────────────────────────────────────────────────────

    • Remote storage URL: https://eradiate-data-registry.s3.eu-west-3.amazonaws.com/registry-v1/
    • Asset cache location [300 MB]: /home/user/.cache/eradiate/cached
    • Unpacked asset location [430 MB]: /home/user/.cache/eradiate/unpacked
    • Installation location: /home/user/.cache/eradiate/installed/eradiate-v0.31.0

    ── File resolver ─────────────────────────────────────────────────────────────────────────────

    • /home/user/.cache/eradiate/installed/eradiate-v0.31.0
    • /home/user/Documents/src/rayference/rtm/eradiate/resources/data

Accessing data
--------------

The file resolver is used in many components to resolve relative paths. This
notably means that:

* users can relocate their data provided that they do not modify the file tree
  and that they make sure that the relocation target directory is added to the
  file resolver;

* developers can rely on the file resolver to look up shipped data using
  relative paths, because the resource installation location is always added to
  the file resolver.
