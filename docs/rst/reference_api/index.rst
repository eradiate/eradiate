.. _sec-reference_api:

API reference
=============

.. button-ref:: sec-reference_plugins
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

Alphabetical list of modules
----------------------------

.. toctree::
   :maxdepth: 1

   eradiate_core
   attrs
   ckd
   converters
   contexts
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
   test_tools
   thermoprops
   units
   validators
   warp
   xarray

Private modules
---------------

.. toctree::
   :maxdepth: 1

   config
   factory
   mode
   presolver
   util
