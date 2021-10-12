import typing as t

import attr

from ._core import Measure
from ._pipeline import PipelineStep

# ------------------------------------------------------------------------------
#                       Post-processing pipeline steps
# ------------------------------------------------------------------------------


@attr.s
class Assemble(PipelineStep):
    """
    Assemble raw kernel results (output as nested dictionaries) into an xarray
    dataset.
    """

    sensor_dims = attr.ib()
    film_dims = attr.ib()

    def transform(self, *args, **kwargs) -> t.Any:
        raise NotImplementedError


@attr.s
class Aggregate(PipelineStep):
    def transform(self, *args, **kwargs) -> t.Any:
        raise NotImplementedError


@attr.s
class MultiDistantMeasure(Measure):
    post_processing_pipeline = attr.ib(
        default=[
            (
                "assemble",
                Assemble(sensor_dims=("spp",), film_dims=("x", "y")),
            ),
            ("aggregate", Aggregate()),
            ("map_viewing_angles", MapViewingAngles()),
            ("add_illumination", AddIllumination()),
            ("compute_reflectance", ComputeReflectance()),
        ]
    )
