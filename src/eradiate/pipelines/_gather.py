from __future__ import annotations

import logging

import attrs
import numpy as np
import xarray as xr
from pinttr.util import always_iterable

import eradiate

from ._core import PipelineStep
from ..attrs import documented, parse_docs
from ..exceptions import UnsupportedModeError
from ..kernel import bitmap_to_dataset
from ..units import symbol
from ..units import unit_context_config as ucc

logger = logging.getLogger(__name__)


def _spectral_dims():
    if eradiate.mode().is_mono:
        return (
            (
                "w",
                {
                    "standard_name": "radiation_wavelength",
                    "long_name": "wavelength",
                    "units": symbol(ucc.get("wavelength")),
                },
            ),
        )
    elif eradiate.mode().is_ckd:
        return (
            ("bin", {"standard_name": "ckd_bin", "long_name": "CKD bin"}),
            ("index", {"standard_name": "ckd_index", "long_name": "CKD index"}),
        )
    else:
        raise UnsupportedModeError


@parse_docs
@attrs.define
class Gather(PipelineStep):
    """
    Gather raw kernel results (output as nested dictionaries) into an xarray
    dataset.

    This pipeline step takes a nested dictionary produced by the parametric loop
    of an :class:`.Experiment` and repackages it as a :class:`~xarray.Dataset`.
    The top-level spectral index is mapped to mode-dependent spectral
    coordinates. Film dimensions are left unmodified and retain their
    metadata.

    An ``img`` variable holds sensor values. An ``spp`` variable holds the
    sample count.
    """

    var: str | tuple[str, dict] = documented(
        attrs.field(default="img"),
        default='"img"',
        type="str or tuple[str, dict]",
        init_type="str or tuple[str, dict], optional",
        doc="Name of the variable containing sensor data. Optionally, a "
        "(name, metadata) pair can be passed.",
    )

    def transform(self, x: dict) -> xr.Dataset:
        logger.debug("gather: begin")
        # Basic preparation
        spectral_dims = []
        spectral_dim_metadata = {}

        for y in _spectral_dims():
            if isinstance(y, str):
                spectral_dims.append(y)
                spectral_dim_metadata[y] = {}
            else:
                spectral_dims.append(y[0])
                spectral_dim_metadata[y[0]] = y[1]

        sensor_datasets = []

        # Loop on spectral indexes
        for siah, result_dict in x.items():
            if eradiate.mode().is_mono:
                spectral_index = siah
            elif eradiate.mode().is_ckd:
                spectral_index = (
                    str(int(siah[0])),  # TODO: PR#311 hack
                    siah[1],
                )

            ds = bitmap_to_dataset(result_dict["bitmap"])
            spp = result_dict["spp"]

            # Set spectral coordinates
            all_coords = {
                spectral_dim: [spectral_coord]
                for spectral_dim, spectral_coord in zip(
                    spectral_dims, always_iterable(spectral_index)
                )
            }

            # Add spectral and sensor dimensions to img array
            ds["img"] = ds.img.expand_dims(dim=all_coords)

            # Package spp in a data array
            all_dims = list(all_coords.keys())
            ds["spp"] = (all_dims, np.reshape(spp, [1 for _ in all_dims]))

            sensor_datasets.append(ds)

        # Combine all the data
        with xr.set_options(keep_attrs=True):
            result = xr.combine_by_coords(sensor_datasets)

        # Drop "channel" dimension when using a monochromatic Mitsuba variant
        if eradiate.mode().check(mi_color_mode="mono"):
            result = result.squeeze("channel", drop=True)

        for spectral_dim in spectral_dims:
            result[spectral_dim].attrs = spectral_dim_metadata[spectral_dim]

        # Apply metadata to data variables
        if isinstance(self.var, str):
            var = self.var
            var_metadata = {}
        else:
            var = self.var[0]
            var_metadata = self.var[1]

        result = result.rename({"img": var})
        result[var].attrs.update(var_metadata)

        logger.debug("gather: end")

        return result
