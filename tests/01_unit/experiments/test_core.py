import pytest
import xarray as xr

import eradiate.kernel.logging
from eradiate.experiments import AtmosphereExperiment
from eradiate.units import unit_registry as ureg

eradiate.kernel.install_logging()


@pytest.mark.parametrize("measures", [None, 1, [0, 1]])
def test_run_function(modes_all_double, measures):
    srf = {"type": "delta", "wavelengths": [540, 550] * ureg.nm}

    exp = AtmosphereExperiment(
        atmosphere=None,
        measures=[
            {"type": "mdistant", "id": f"mdistant_{i}", "srf": srf} for i in range(3)
        ],
    )
    result = eradiate.run(exp)
    assert isinstance(result, xr.Dataset if measures == 1 else dict)
