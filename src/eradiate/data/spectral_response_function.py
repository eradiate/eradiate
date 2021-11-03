"""
Spectral response function data getter.
"""
import enum

import xarray as xr

from .core import DataGetter
from .._presolver import path_resolver as _presolver


class Platform(enum.Enum):
    """
    Platform enumeration.
    """

    Sentinel_2A = "sentinel_2a"
    Sentinel_2B = "sentinel_2b"
    Sentinel_3A = "sentinel_3a"
    Sentinel_3B = "sentinel_3b"
    Terra = "terra"


class Instrument(enum.Enum):
    """
    Instrument enumeration.
    """

    MODIS = "modis"
    MSI = "msi"
    OLCI = "olci"
    SLSTR = "slstr"


_MODIS_BANDS = [x for x in range(1, 37)]
_MSI_BANDS = [1, 2, 3, 4, 5, 6, 7, 8, "8a", 9, 10, 11, 12]
_OLCI_BANDS = [x for x in range(1, 22)]
_SLSTR_BANDS = [x for x in range(1, 10)]

_BANDS = {
    (Platform.Sentinel_2A, Instrument.MSI): _MSI_BANDS,
    (Platform.Sentinel_2B, Instrument.MSI): _MSI_BANDS,
    (Platform.Sentinel_3A, Instrument.OLCI): _OLCI_BANDS,
    (Platform.Sentinel_3A, Instrument.SLSTR): _SLSTR_BANDS,
    (Platform.Sentinel_3B, Instrument.OLCI): _OLCI_BANDS,
    (Platform.Sentinel_3B, Instrument.SLSTR): _SLSTR_BANDS,
    (Platform.Terra, Instrument.MODIS): _MODIS_BANDS,
}


def _srf_ds_id(platform, instrument, band):
    platform_id = platform.value.lower()
    instrument_id = instrument.value.lower()
    band_id = str(band).lower()
    return f"{platform_id}-{instrument_id}-{band_id}"


_PATHS = {}
for platform, instrument in _BANDS:
    for band in _BANDS[(platform, instrument)]:
        ds_id = _srf_ds_id(platform, instrument, band)
        _PATHS[ds_id] = f"spectra/srf/{ds_id}.nc"


class _SpectralResponseFunctionGetter(DataGetter):
    PATHS = _PATHS

    @classmethod
    def open(cls, id):
        path = _presolver.resolve(cls.path(id))
        return xr.open_dataset(path)

    @classmethod
    def find(cls):
        result = {}

        for id in cls.PATHS.keys():
            path = _presolver.resolve(cls.path(id))
            result[id] = path.is_file()

        return result
