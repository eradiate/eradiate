.. _sec-reference-xarray:

Dataset and array management [eradiate.xarray]
==============================================

Creation [eradiate.xarray.make]
-------------------------------
.. currentmodule:: eradiate.xarray.make

.. autosummary::
   :toctree: generated/

   make_dataarray

Metadata specification and validation [eradiate.xarray.metadata]
----------------------------------------------------------------
.. currentmodule:: eradiate.xarray.metadata

.. autosummary::
   :toctree: generated/

   validate_metadata
   DataSpec
   CoordSpec
   VarSpec
   DatasetSpec
   CoordSpecRegistry

Data selection [eradiate.xarray.select]
---------------------------------------
.. currentmodule:: eradiate.xarray.select

.. autosummary::
   :toctree: generated/

   plane
   pplane

Private: Accessors [eradiate.xarray._accessors]
-----------------------------------------------
.. currentmodule:: eradiate.xarray._accessors

.. autosummary::
   :toctree: generated/

   EradiateDataArrayAccessor
   EradiateDatasetAccessor
