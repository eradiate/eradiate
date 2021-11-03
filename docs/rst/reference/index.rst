.. _sec-reference:

API reference
=============

Eradiate's API reference documentation is generated automatically using Sphinx's
``autodoc`` and ``autosummary`` extensions.

.. note:: In addition to APIs, meant to be public and used both externally
   (by end-users) and internally (by maintainers), this reference manual also
   documents internal entry points (used by maintainers within Eradiate).
   Private components are contained in underscore-prefixed modules. In order to
   become part of the API, they need to be either transferred to a public module
   or exposed in a public module.

.. toctree::
   :maxdepth: 2

   ckd
   class_writing
   config
   contexts
   core
   data
   exceptions
   experiments
   factory
   kernel
   math
   misc
   notebook
   pipelines
   plot
   radprops
   scenes
   solvers
   thermoprops
   units
   xarray
