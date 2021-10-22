from ._aggregate import AggregateSampleCount
from ._assemble import AddViewingAngles
from ._core import Pipeline
from ._gather import Gather
from ..scenes.measure import Measure
from ..units import unit_context_kernel as uck


def default_pipeline(measure: Measure):
    """
    Generate a default post-processing pipeline. It includes the following
    steps:

    1. :class:`.Gather`
    2. :class:`.AggregateSampleCount`
    3. :class:`.AddViewingAngles`

    Returns
    -------
    :class:`.Pipeline`
    """
    assemble = Gather(
        sensor_dims=["spp"] if measure.split_spp else [],
        img_var=(
            "lo",
            {
                "units": uck.get("radiance"),
                "standard_name": "leaving_radiance",
                "long_name": "leaving radiance",
            },
        ),
    )
    add_viewing_angles = AddViewingAngles(multi_distant=measure)

    return Pipeline(
        [
            ("assemble", assemble),
            ("aggregate_sample_count", AggregateSampleCount()),
            ("add_viewing_angles", add_viewing_angles),
            # ("add_illumination", AddIllumination()),
            # ("compute_reflectance", ComputeReflectance()),
        ]
    )
