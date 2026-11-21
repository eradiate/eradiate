"""
Tests for behaviour common to multiple experiment classes.
"""

import pytest

from eradiate import unit_registry as ureg
from eradiate.experiments import (
    AtmosphereExperiment,
    CanopyAtmosphereExperiment,
    CanopyExperiment,
    DEMExperiment,
)


class TestExperimentCommon:
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
    def test_geometry_bounds(
        self, mode_mono, cls, ground_altitude, toa_altitude, expected
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

    @pytest.mark.parametrize(
        "cls",
        [
            AtmosphereExperiment,
            CanopyExperiment,
            CanopyAtmosphereExperiment,
            DEMExperiment,
        ],
        ids=["atmosphere", "canopy", "canopy_atmosphere", "dem"],
    )
    def test_no_illumination(self, mode_mono, cls):
        """
        Setting illumination to None inserts no emitter in the kernel scene.
        """
        # Rendering is deliberately not tested: an emitter-free scene is
        # pointless and fails to render with volpath in JIT variants.
        kwargs = {"illumination": None, "measures": {"type": "mdistant"}}
        if cls is not CanopyExperiment:
            kwargs["atmosphere"] = None
        exp = cls(**kwargs)
        assert "illumination" not in exp.scene.objects

        exp.init()
        assert len(exp.mi_scene.obj.emitters()) == 0
