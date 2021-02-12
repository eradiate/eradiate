"""The Eradiate radiative transfer simulation software package."""

__version__ = "0.0.1"  #: Eradiate version number.

# -- Operational mode definition -----------------------------------------------

from ._mode import mode, set_mode, modes

# -- xarray accessor imports ---------------------------------------------------

from .util.xarray import EradiateDataArrayAccessor, EradiateDatasetAccessor
del EradiateDataArrayAccessor, EradiateDatasetAccessor
