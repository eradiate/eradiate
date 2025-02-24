import pytest
import xarray as xr

import eradiate.kernel.logging
from eradiate.experiments import AtmosphereExperiment
from eradiate.units import unit_registry as ureg

eradiate.kernel.install_logging()


@pytest.mark.parametrize(
    "measures, expected_type", [(None, dict), (1, xr.Dataset), ([0, 1], dict)]
)
def test_run_function(modes_all_double, measures, expected_type):
    srf = {"type": "delta", "wavelengths": [540, 550] * ureg.nm}

    exp = AtmosphereExperiment(
        atmosphere=None,
        measures=[
            {"type": "mdistant", "id": f"mdistant_{i}", "srf": srf} for i in range(3)
        ],
    )
    result = eradiate.run(exp, measures=measures)
    assert isinstance(result, expected_type)
