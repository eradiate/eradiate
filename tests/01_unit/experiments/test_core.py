import xarray as xr

from eradiate.experiments import AtmosphereExperiment
from eradiate.units import unit_registry as ureg


def test_run_function(modes_all_double):
    measure = {
        "type": "mdistant",
        "srf": {
            "type": "multi_delta",
            "wavelengths": [540, 550] * ureg.nm,
        },
    }

    exp = AtmosphereExperiment(atmosphere=None, measures=measure)
    result = eradiate.run(exp)
    assert isinstance(result, xr.Dataset)
