import logging
import re
import typing as t

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
                    "standard_name": "wavelength",
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

    This pipeline step takes a nested dictionary produced by the spectral loop
    of an :class:`.Experiment` and repackages it as a :class:`~xarray.Dataset`.
    The top-level spectral index is mapped to mode-dependent spectral
    coordinates. Results from the sensors part of a single measurement are
    mapped to specified sensor dimensions, which can then be further processed
    by aggregation steps. Film dimensions are left unmodified and retain their
    metadata.

    An ``img`` variable holds sensor values. An ``spp`` variable holds sample
    count.
    """

    prefix: str = documented(
        attrs.field(default=r".*"),
        default='".*"',
        type="str",
        init_type="str, optional",
        doc="Prefix string used to match sensor IDs. The default value will "
        "match anything.",
    )

    sensor_dims: t.Sequence = documented(
        attrs.field(
            factory=list,
            validator=attrs.validators.instance_of((list, tuple)),
        ),
        default="[]",
        type="list of (str or tuple)",
        init_type="list of (str or tuple), optional",
        doc="List of sensor dimensions. Each list item can be a string or a "
        "(dimension, metadata) pair.",
    )

    var: t.Union[str, t.Tuple[str, t.Dict]] = documented(
        attrs.field(default="img"),
        default='"img"',
        type="str or tuple[str, dict]",
        init_type="str or tuple[str, dict], optional",
        doc="Name of the variable containing sensor data. Optionally, a "
        "(name, metadata) pair can be passed.",
    )

    def transform(self, x: t.Dict) -> xr.Dataset:
        # Basic preparation
        prefix = self.prefix

        sensor_dims = []
        sensor_dim_metadata = {}

        for y in self.sensor_dims:
            if isinstance(y, str):
                sensor_dims.append(y)
                sensor_dim_metadata[y] = {}
            else:
                sensor_dims.append(y[0])
                sensor_dim_metadata[y[0]] = y[1]

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

        regex = re.compile(
            r"\_".join(
                [prefix]
                + [rf"{sensor_dim}(?P<{sensor_dim}>\d*)" for sensor_dim in sensor_dims]
            )
        )

        # Loop on spectral indexes
        for spectral_index in x.keys():
            # Loop on sensors
            for sensor_id, sensor_bitmap in x[spectral_index]["values"].items():
                # Collect data
                ds = bitmap_to_dataset(sensor_bitmap)
                spp = x[spectral_index]["spp"][sensor_id]

                # Set spectral coordinates
                spectral_coords = {
                    spectral_dim: [spectral_coord]
                    for spectral_dim, spectral_coord in zip(
                        spectral_dims, always_iterable(spectral_index)
                    )
                }

                # Detect sensor coordinates
                match = regex.match(sensor_id)

                if match is None:
                    raise RuntimeError(
                        "could not detect requested sensor dimensions in "
                        f"sensor ID '{sensor_id}' using regex '{regex.pattern}'; "
                        "this could be due to incorrect values or order of "
                        "'sensor_dims'"
                    )

                sensor_coords = {
                    f"{sensor_dim}_index": [int(match.group(sensor_dim))]
                    for sensor_dim in sensor_dims
                }

                # Add spp dimension even though sample count split did not
                # produce any extra sensor
                if "spp" not in sensor_dims:
                    sensor_coords["spp_index"] = [0]

                # Add spectral and sensor dimensions to img array
                all_coords = {**spectral_coords, **sensor_coords}
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

        # Apply metadata to new dimensions
        for sensor_dim in sensor_dims:
            result[f"{sensor_dim}_index"].attrs = sensor_dim_metadata[sensor_dim]

        if "spp" not in sensor_dims:
            result["spp_index"].attrs = {
                "standard_name": "spp_index",
                "long_name": "SPP index",
            }

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

        result["spp"].attrs = {
            "standard_name": "sample_count",
            "long_name": "sample count",
        }

        return result
