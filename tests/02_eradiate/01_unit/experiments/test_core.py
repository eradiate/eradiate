import xarray as xr

import eradiate.kernel.logging
from eradiate.experiments import AtmosphereExperiment
from eradiate.units import unit_registry as ureg

eradiate.kernel.install_logging()


def test_run_function(modes_all_double):
    mode = eradiate.mode()

    measure = {"type": "mdistant"}
    measure["srf"] = {
        "type": "multi_delta",
        "wavelengths": [540, 550] * ureg.nm,
    }

    exp = AtmosphereExperiment(atmosphere=None, measures=measure)
    result = eradiate.run(exp)
    assert isinstance(result, xr.Dataset)
