..
  This file documents imported variables. We must do this because the
  ``autodoc`` Sphinx extension cannot collect their docstrings.

eradiate.unit_registry
======================

.. data:: eradiate.unit_registry
   :annotation: = pint.UnitRegistry()

   Unit registry common to all Eradiate components. All units used in Eradiate
   must be created using this registry.

   .. seealso:: :class:`pint.UnitRegistry`
