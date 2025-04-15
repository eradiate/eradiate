import pytest
import xarray as xr

import eradiate
from eradiate.experiments import AtmosphereExperiment
from eradiate.experiments._core import MeasureRegistry
from eradiate.units import unit_registry as ureg


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
    "measures, expected_type", [(None, dict), (1, xr.Dataset), ([0, 1], dict)]
)
def test_run_function(modes_all_double, srf, measures, expected_type):
    exp = AtmosphereExperiment(
        atmosphere=None,
        measures=[
            {"type": "mdistant", "id": f"mdistant_{i}", "srf": srf} for i in range(3)
        ],
    )
    result = eradiate.run(exp, measures=measures)
    assert isinstance(result, expected_type)
