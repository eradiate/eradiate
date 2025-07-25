.. _sec-reference_api:

API reference
=============

.. button-ref:: /reference_plugins/index
   :ref-type: doc
   :color: primary
   :expand:

   Looking for the plugin reference? Click here!

Eradiate's API reference documentation is generated automatically using Sphinx's
``autodoc`` and ``autosummary`` extensions.

.. note:: In addition to APIs, meant to be public and used both externally
   (by end-users) and internally (by maintainers), this reference manual also
   documents internal entry points (used by maintainers within Eradiate).
   Private components are contained in underscore-prefixed modules. In order to
   become part of the API, they need to be either transferred to a public module
   or exposed in a public module.

Quick access
------------

.. grid:: 1 2 auto auto
    :gutter: 3

    .. grid-item-card:: :iconify:`material-symbols:play-circle height=1.5em` Core
        :link: eradiate_core
        :link-type: doc

        ``eradiate``

    .. grid-item-card:: :iconify:`material-symbols:public height=1.5em` Scenes
        :link: scenes
        :link-type: doc

        ``eradiate.scenes``

    .. grid-item-card:: :iconify:`material-symbols:science height=1.5em` Experiments
        :link: experiments
        :link-type: doc

        ``eradiate.experiments``

    .. grid-item-card:: :iconify:`material-symbols:settings height=1.5em` Configuration
        :link: config
        :link-type: doc

        ``eradiate.config``

Alphabetical list of modules
----------------------------

.. toctree::
   :maxdepth: 1

   eradiate_core
   attrs
   config
   constants
   contexts
   converters
   data
   exceptions
   experiments
   frame
   kernel
   notebook
   pipelines
   plot
   quad
   radprops
   rng
   scenes
   spectral
   srf_tools
   test_tools
   units
   util
   validators
   warp
   xarray

Private modules
---------------

.. toctree::
   :maxdepth: 1

   factory
   mode
