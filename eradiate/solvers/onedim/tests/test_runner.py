import numpy as np
import pytest

import eradiate.kernel
from eradiate.scenes.core import KernelDict
from eradiate.scenes.measure import RadianceMeterHsphereMeasure
from eradiate.solvers.onedim.runner import OneDimRunner
from eradiate.exceptions import KernelVariantError


def test_onedim_runner_construct(mode_mono):
    # Instantiate default runner (empty kernel dict)
    runner = OneDimRunner()
    assert runner.kernel_dict == KernelDict.empty()

    # Instantiate from dictionary
    runner = OneDimRunner({})
    assert isinstance(runner.kernel_dict, KernelDict)
    assert runner.kernel_dict == {}


def test_onedim_runner_run(mode_mono):
    # Basic configuration
    vza = np.arange(0, 90, 30)  # Viewing zenith
    vaa = np.arange(0, 360, 180)  # Viewing azimuth
    rho = 0.5  # Surface reflectivity
    spp = 1  # Number of samples per measure
    li = 1.  # Incoming irradiance

    # Instantiate default runner (empty kernel dict)
    runner = OneDimRunner()

    # Add surface manually
    runner.kernel_dict["surface"] = {
        "type": "rectangle",
        "bsdf": {
            "type": "diffuse",
            "reflectance": {"type": "uniform", "value": rho}
        }
    }

    # Add illumination manually
    runner.kernel_dict["illumination"] = {
        "type": "directional",
        "direction": [0, 0, -1],
        "irradiance": {"type": "uniform", "value": li}
    }

    # Use scene element to add measure
    measure = RadianceMeterHsphereMeasure(
        zenith_res=30., azimuth_res=180., direction=[0, 0, -1],
        origin=[0, 0, 0.01], spp=spp
    )
    runner.kernel_dict.add(measure)

    theoretical_solution = np.full((len(vza) * len(vaa)), rho * li / np.pi)

    result = runner.run()
    assert np.allclose(result["radiancemeter_hsphere"], theoretical_solution)

    # Check that run attempt with wrong variant raises
    with pytest.raises(KernelVariantError):
        eradiate.kernel.set_variant("scalar_rgb")
        runner.run()
