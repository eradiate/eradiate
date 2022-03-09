.. _sec-developer_guide-data_guide:

Data guide
==========

This guide describes how the data shipped with Eradiate is managed.

Overview
--------

Eradiate ships data of various sizes and maturity level. The current data
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
------------------

Each store requires a different protocol.

``small_files``
    Install `pre-commit <https://pre-commit.com/>`_ and activate the hooks, this
    will update the registry automatically upon commit. The rules used to create
    the registry file are defined in the ``"registry_rules.yml"`` file. If, for
    some reason, you cannot use pre-commit, then you must be very careful and
    update the registry manually using the ``ertdata make-registry``
    command-line tool (it should be run in the data submodule).

``large_files_stable``
    The most complicated: avoid updating the files, just add new ones. When
    doing so, you have to update the registry: compute the sha256 hash of the
    new file (*e.g.* ``sha256sum`` command-line tool) and update the registry
    file with this new entry. If you happen to have the full contents of the
    data store on your hard drive, you may also use the
    ``ertdata make-registry`` command-line tool to update the registry
    automatically.

``large_files_unstable``
    The simplest: just drop the file in the remote storage, it will be
    immediately accessible.
