import re
import typing as t

import attr
import numpy as np
import xarray as xr
from pinttr.util import always_iterable

import eradiate

from ._core import Measure
from ._pipeline import PipelineStep
from ..._mode import ModeFlags
from ...attrs import documented, parse_docs
from ...exceptions import UnsupportedModeError

# ------------------------------------------------------------------------------
#                       Post-processing pipeline steps
# ------------------------------------------------------------------------------


def _spectral_dims():
    if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
        return ("w",)
    elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
        return ("bin", "index")
    else:
        raise UnsupportedModeError


@attr.s
class Assemble(PipelineStep):
    """
    Assemble raw kernel results (output as nested dictionaries) into an xarray
    dataset.

    This pipeline step takes a nested dictionary produced by the spectral loop
    of an :class:`.Experiment` and repackages it as a :class:`~xarray.Dataset`.
    The top-level spectral index is mapped to mode-dependent spectral
    coordinates. Results from the sensors part of a single measurement are
    mapped to specified sensor dimensions, which can then be further processed
    by aggregation steps. Film dimensions are left unmodified and retain their
    metadata.

    An ``img`` variable hold sensor results; a ``spp`` variable holds sample
    count if ``"spp"`` is part of the requested sensor dimensions.
    """

    prefix = attr.ib(default=r".*")
    sensor_dims = attr.ib(
        default=[],
        validator=attr.validators.instance_of((list, tuple)),
    )

    def transform(self, x: t.Dict) -> xr.Dataset:
        # Basic preparation
        prefix = self.prefix
        sensor_dims = self.sensor_dims
        spectral_dims = _spectral_dims()
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
            for sensor_id, sensor_data in x[spectral_index]["values"].items():
                # Collect data
                ds = sensor_data.copy(deep=False)
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
                    spectral_coords["spp_index"] = [0]

                # Add spectral and sensor dimensions to img array
                ds["img"] = ds.img.expand_dims(dim={**spectral_coords, **sensor_coords})

                # Package spp in a data array
                ds["spp"] = ("spp_index", [spp])

                sensor_datasets.append(ds)

        # Combine all the data
        return xr.merge(sensor_datasets)


@attr.s
class AggregateSampleCount(PipelineStep):
    def transform(self, x: t.Any) -> t.Any:
        raise NotImplementedError


@parse_docs
@attr.s
class MultiDistantMeasure(Measure):
    directions: np.ndarray = documented(
        attr.ib(
            default=np.array([[0.0, 0.0, -1.0]]),
            converter=np.array,
        ),
        doc="A sequence of 3-vectors specifying distant sensing directions.",
        type="ndarray",
        init_type="array-like",
        default="[[0, 0, -1]]",
    )

    post_processing_pipeline = attr.ib(
        default=[
            ("assemble", Assemble(sensor_dims=("spp",))),
            ("aggregate_sample_count", AggregateSampleCount()),
            # ("map_viewing_angles", MapViewingAngles()),
            # ("add_illumination", AddIllumination()),
            # ("compute_reflectance", ComputeReflectance()),
        ]
    )

    def sensor_coords(self):
        """
        Compute sensor coordinates.
        """
        return {"spp": ("spp", self._split_spps(), {"long_name": "sample count"})}

    def sensor_id(self, i_spp=None):
        """
        Assemble a sensor ID from indexes on sensor coordinates.
        """
        components = [self.id]

        if i_spp is not None:
            components.append(f"spp{i_spp}")

        return "_".join(components)
