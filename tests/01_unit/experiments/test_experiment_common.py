"""
Tests for behaviour common to multiple experiment classes.
"""

import pytest

from eradiate import unit_registry as ureg
from eradiate.experiments import (
    AtmosphereExperiment,
    CanopyAtmosphereExperiment,
    DEMExperiment,
)


@pytest.mark.parametrize(
    "cls",
    [
        AtmosphereExperiment,
        CanopyAtmosphereExperiment,
        DEMExperiment,
    ],
    ids=[
        "atmosphere",
        "canopy_atmosphere",
        "dem",
    ],
)
@pytest.mark.parametrize(
    "ground_altitude, toa_altitude, expected",
    [
        (0, 120, "pass"),
        (-1, 120, "raises"),
        (0, 121, "raises"),
        (-1, 121, "raises"),
    ],
)
def test_atmosphere_experiment_geometry_bounds(
    mode_mono, cls, ground_altitude, toa_altitude, expected
):
    """
    Incompatible geometry bounds and atmosphere extent raise an exception.
    """
    kwargs = {
        "atmosphere": {"type": "molecular"},
        "geometry": {
            "type": "plane_parallel",
            "ground_altitude": ground_altitude * ureg.km,
            "toa_altitude": toa_altitude * ureg.km,
        },
    }
    if expected == "raises":
        with pytest.raises(ValueError):
            cls(**kwargs)
    else:
        cls(**kwargs)
