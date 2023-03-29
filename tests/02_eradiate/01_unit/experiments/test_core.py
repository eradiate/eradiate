import xarray as xr

import eradiate.kernel.logging
from eradiate.experiments import AtmosphereExperiment

eradiate.kernel.install_logging()


def test_run_function(modes_all_double):
    mode = eradiate.mode()

    measure = {"type": "mdistant"}
    if mode.is_mono:
        measure["spectral_cfg"] = {"wavelengths": [540.0, 550.0]}

    elif mode.is_ckd:
        measure["spectral_cfg"] = {"bin_set": "10nm", "bins": ["540", "550"]}

    else:
        assert False, f"Please add a test for mode '{mode.id}'"

    exp = AtmosphereExperiment(atmosphere=None, measures=measure)
    result = eradiate.run(exp)
    assert isinstance(result, xr.Dataset)
