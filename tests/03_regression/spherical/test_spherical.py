import numpy as np
import pytest

import eradiate
from eradiate import fresolver
from eradiate import unit_registry as ureg
from eradiate.constants import EARTH_RADIUS
from eradiate.experiments import AtmosphereExperiment
from eradiate.test_tools.regression import SidakTTest


@pytest.mark.regression
@pytest.mark.slow
def test_spherical(mode_ckd_double, artefact_dir, plot_figures):
    spp = 100
    config = {
        "geometry": "spherical_shell",
        "surface": {
            "type": "rpv",
            "rho_0": {"type": "uniform", "value": 0.017051},
            "k": {"type": "uniform", "value": 0.95},
            "g": {"type": "uniform", "value": -0.1},
            "rho_c": {"type": "uniform", "value": 0.017051},
        },
        "atmosphere": {
            "type": "molecular",
            "has_absorption": True,
            "has_scattering": True,
            "thermoprops": {
                "identifier": "afgl_1986-us_standard",
                "z": np.arange(0, 120.05, 0.05) * ureg.km,
            },
            "absorption_data": "monotropa",
        },
        "illumination": {
            "type": "directional",
            "zenith": 30.0 * ureg.deg,
            "azimuth": 0.0 * ureg.deg,
        },
        "measures": [
            {
                "type": "mdistant",
                "construct": "hplane",
                "zeniths": np.arange(-85.0, 65.0, 1.0) * ureg.deg,
                "azimuth": 0.0 * ureg.deg,
                "srf": "sentinel_2a-msi-4",
                "spp": spp,
                "target": [0.0, 0.0, EARTH_RADIUS.m_as("km")] * ureg.km,
            }
        ],
        "ckd_quad_config": {"type": "gauss_legendre", "ng_max": 16, "policy": "fixed"},
        "integrator": {"type": "volpath", "moment": True},
    }

    exp = AtmosphereExperiment(**config)
    result = eradiate.run(exp)
    reference = fresolver.load_dataset(
        "tests/regression_test_references/test_spherical_shell-ref.nc"
    )

    test = SidakTTest(
        name="test_spherical_shell",
        value=result,
        reference=reference,
        threshold=0.01,
        archive_dir=artefact_dir,
        variable="radiance",
        plot=False,
    )

    assert test.run(plot_figures)
