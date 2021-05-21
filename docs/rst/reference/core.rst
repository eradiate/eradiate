.. _sec-reference-core:

Core support components
=======================

Mode control
------------
.. currentmodule:: eradiate

.. autosummary::
   :toctree: generated/

   mode
   modes
   set_mode

.. dropdown:: **Private: Mode implementation details**

   .. autosummary::
      :toctree: generated/

      _mode.ModeSpectrum
      _mode.ModePrecision
      _mode.Mode

Units and quantities
--------------------
.. currentmodule:: eradiate

.. toctree::
   :hidden:
   :maxdepth: 1

   imported_vars/unit_registry
   imported_vars/unit_context_config
   imported_vars/unit_context_kernel

.. list-table::
   :widths: 25 75

   * - :data:`unit_registry`
     - Unit registry common to all Eradiate components.
   * - :data:`unit_context_config`
     - Unit context used when interpreting configuration dictionaries.
   * - :data:`unit_context_kernel`
     - Unit context used when building kernel dictionaries.

.. dropdown:: **Private: Physical quantity enum**

   .. autosummary::
      :toctree: generated

       _units.PhysicalQuantity

.. dropdown:: **Private: Quantity converter**

   .. autosummary::
      :toctree: generated/
      
      _units.to_quantity  

Path resolver
-------------
.. currentmodule:: eradiate

.. toctree::
   :hidden:
   :maxdepth: 1

   imported_vars/path_resolver

.. list-table::
   :widths: 25 75

   * - :data:`path_resolver`
     - Unique path resolver instance.

.. dropdown:: **Private: Path resolver implementation**

   .. autosummary::
      :toctree: generated

       _presolver.PathResolver

Miscellaneous
-------------
.. currentmodule:: eradiate

.. autosummary::
   :toctree: generated/

   __version__
