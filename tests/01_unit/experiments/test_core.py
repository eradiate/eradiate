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
        return {measure.id: measure for measure in self.measures}


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

    # Check initialization context
    assert np.allclose(exp.context_init().si.w.m, 440.0)

    # We need to init the scene so that mapping measures to sensors is possible
    exp.init()

    # Check contexts generated for each measure configuration
    for measures, expected in [
        (None, [(440.0, [0, 2]), (550.0, [1, 2, 3]), (660.0, [3])]),
        (0, [(440.0, [0])]),
        ([0, 1], [(440.0, [0]), (550.0, [1])]),
        ([0, 1, 2], [(440.0, [0, 2]), (550.0, [1, 2])]),
        ([1, 3], [(550.0, [1, 3]), (660.0, [3])]),
        ([2, 3], [(440.0, [2]), (550.0, [2, 3]), (660.0, [3])]),
    ]:
        result = exp.contexts(measures)
        assert len(result) == len(expected)
        for ctx, (w, sensors) in zip(result, expected):
            assert np.isclose(ctx.si.w.m, w), f"{measures = }, {w = }"
            assert ctx.active_sensors == sensors, f"{measures = }, {w = }"


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
            [{"type": "mdistant", "id": "mdistant", "srf": srf} for _ in range(3)]
        )


@pytest.fixture(scope="function")
def atmosphere_experiment():
    yield AtmosphereExperiment(
        atmosphere=None,
        measures=[
            {
                "type": "mdistant",
                "id": f"mdistant_{i}",
                "srf": {"type": "delta", "wavelengths": [550.0]},
            }
            for i in range(3)
        ],
    )


@pytest.mark.parametrize(
    "measures, expected_type",
    [
        (None, dict),
        (1, xr.Dataset),
        ([0, 1], dict),
    ],
)
def test_run_function(modes_all_double, atmosphere_experiment, measures, expected_type):
    result = eradiate.run(atmosphere_experiment, measures=measures, spp=4)
    assert isinstance(result, expected_type)


def test_run_function_multiple_times(mode_mono, atmosphere_experiment):
    result = eradiate.run(atmosphere_experiment, measures=0, spp=4)
    assert isinstance(result, xr.Dataset)
    result = eradiate.run(atmosphere_experiment, measures=1, spp=4)
    assert isinstance(result, xr.Dataset)
    assert len(atmosphere_experiment.results) == 2
