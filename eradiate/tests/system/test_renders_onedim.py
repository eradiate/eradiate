"""Rendering based tests for OneDimSolverApp"""

import pytest
from .util import aov_to_variance, z_test
from eradiate.solvers.onedim.app import OneDimSolverApp

import matplotlib.pyplot as plt
import numpy as np

def app_config(surface="lambertian", atmosphere=None, illumination="directional", spp=1000):
    """Return a valid config file for the OneDimSolverApp.
    To set constant illumination, set the illumination_direction to [0, 0, 0].
    """
    config = {}

    # create surface config section
    if surface == "lambertian":
        config["surface"] = {
            "type": "lambertian",
            "reflectance": {
                "type": "uniform",
                "value": 0.5
            }
        }
    elif surface == "rpv":
        config["surface"] = {
            "type": "rpv",
            "rho_0": {
                "type": "uniform",
                "value": 0.183
            },
            "k": {
                "type": "uniform",
                "value": 0.780
            },
            "ttheta": {
                "type": "uniform",
                "value": -0.1
            }
        }

    # create atmosphere config section
    if atmosphere is None:
        pass
    elif atmosphere == "rayleigh":
        config["atmosphere"] = {
        "type": "rayleigh_homogeneous",
        "height": 120.,
        "height_units": "km",
        "sigma_s": 1.e-4
    }

    # illumination config section
    if illumination=="constant":
        config["illumination"] = {
            "type": "constant",
            "radiance": {
                "type": "uniform",
                "value": 1e3
            }
        }
    elif illumination == "directional":
        config["illumination"] = {
        "type": "directional",
        "zenith": 30.,
        "azimuth": 0.,
        "irradiance": {
            "type": "uniform",
            "value": 1.8e+6,
            "value_units": "W/km**2/nm"
        },
    }

    # measure section
    config["measure"] = [{
        "type": "toa_hsphere",
        "spp": spp,
        "zenith_res": 5.,
        "azimuth_res": 5.
    }]
    return config


@pytest.mark.parametrize("surface", ["lambertian", "rpv"])
@pytest.mark.parametrize("atmosphere", [None, "rayleigh"])
@pytest.mark.parametrize("illumination", ["constant", "directional"])
@pytest.mark.slow
def test_render_onedim(surface, atmosphere, illumination):

    bmp = []
    for spp in [10**i for i in range(5)]:
        config = app_config(surface, atmosphere, illumination, spp)

        app = OneDimSolverApp(config)

        integrator_dict = {
            "integrator": {
                "type": "moment",
                "integrator": {
                    "type": "path"
                }
            }
        }
        app._kernel_dict["integrator"] = integrator_dict
        app.run()

        bmp.append(app.results["toa_hsphere"]["lo"])
    return bmp
