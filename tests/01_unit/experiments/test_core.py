from typing import Any

import numpy as np
import pytest
import xarray as xr

import eradiate
from eradiate.experiments import AtmosphereExperiment, EarthObservationExperiment
from eradiate.experiments._core import MeasureRegistry
from eradiate.scenes.core import SceneElement
from eradiate.units import unit_registry as ureg


class ConcreteEarthObservationExperiment(EarthObservationExperiment):
    def _context_kwargs(self) -> dict[str, Any]:
        return {}

    @property
    def scene_objects(self) -> dict[str, SceneElement]:
        return {}


def test_contexts(mode_mono):
    """
    Check if generated kernel contexts account for selected sensors appropriately.
    """
    exp = ConcreteEarthObservationExperiment(
        measures=[
            {
                "type": "mdistant",
                "id": mes_id,
                "srf": {"type": "delta", "wavelengths": w},
            }
            for mes_id, w in [
                ("mes1", [440.0]),
                ("mes2", [550.0]),
                ("mes3", [440.0, 550.0]),
                ("mes4", [550.0, 660.0]),
            ]
        ]
    )

    assert np.allclose(exp.context_init().si.w.m, 440.0)

    for measures, expected in [
        (None, [440.0, 550.0, 660.0]),
        (0, [440.0]),
        ([0, 1], [440.0, 550.0]),
        ([0, 1, 2], [440.0, 550.0]),
        ([1, 3], [550.0, 660.0]),
        ([2, 3], [440.0, 550.0, 660.0]),
    ]:
        result = [ctx.si.w.m for ctx in exp.contexts(measures)]
        assert np.allclose(result, expected), f"{measures = }"


@pytest.fixture()
def srf():
    return {"type": "delta", "wavelengths": [540, 550] * ureg.nm}


def test_measure_registry(mode_mono, srf):
    m = MeasureRegistry(
        [{"type": "mdistant", "id": f"mdistant_{i}", "srf": srf} for i in range(3)]
    )

    assert m.get_id(2) == "mdistant_2"
    assert m.get_id("mdistant_2") == "mdistant_2"
    assert m.get_index(2) == 2
    assert m.get_index("mdistant_2") == 2

    assert m.resolve(2) is m[2]
    assert m.resolve("mdistant_2") is m[2]


def test_measure_registry_fail(mode_mono, srf):
    # Detect duplicate measure IDs
    with pytest.raises(ValueError):
        MeasureRegistry(
            [{"type": "mdistant", "id": "mdistant", "srf": srf} for i in range(3)]
        )


@pytest.mark.parametrize(
    "measures, expected_type",
    [
        (None, dict),
        (1, xr.Dataset),
        ([0, 1], dict),
    ],
)
def test_run_function(modes_all_double, srf, measures, expected_type):
    exp = AtmosphereExperiment(
        atmosphere=None,
        measures=[
            {"type": "mdistant", "id": f"mdistant_{i}", "srf": srf} for i in range(3)
        ],
    )

    result = eradiate.run(exp, measures=measures, spp=4)
    assert isinstance(result, expected_type)


def test_run_function_multiple_times(mode_mono, srf):
    exp = AtmosphereExperiment(
        atmosphere=None,
        measures=[
            {"type": "mdistant", "id": f"mdistant_{i}", "srf": srf} for i in range(2)
        ],
    )

    result = eradiate.run(exp, measures=0, spp=4)
    assert isinstance(result, xr.Dataset)
    result = eradiate.run(exp, measures=1, spp=4)
    assert isinstance(result, xr.Dataset)
    assert len(exp.results) == 2
