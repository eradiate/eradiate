import pytest

import eradiate
from eradiate.solvers.onedim.rayleigh import RayleighSolverApp
from eradiate.util.units import ureg


def test_rayleigh_solver_app():
    # Test default configuration handling
    app = RayleighSolverApp()
    assert app.config == {
        "atmosphere": {"type": "rayleigh_homogeneous"},
        "illumination": {"type": "directional"},
        "measure": {
            "type": "hemispherical",
            "azimuth_res": 10.0,
            "zenith_res": 10.0,
            "spp": 1000
        },
        "mode": {"type": "mono", "wavelength": 550.0},
        "surface": {"type": "lambertian"}
    }

    # Check that the appropriate variant is selected
    assert eradiate.mode.type == "mono"

    # Check that the default scene can be instantiated
    assert app._kernel_dict.load() is not None

    # Pass a well-formed custom configuration object (without an atmosphere)
    config = {
        "mode": {
            "type": "mono",
            "wavelength": 800.
        },
        "illumination": {
            "type": "directional",
            "zenith": 10.,
            "azimuth": 0.,
            "irradiance": {"type": "uniform", "value": 1.}
        },
        "measure": {
            "type": "hemispherical",
            "zenith_res": 5.,
            "azimuth_res": 10.,
            "spp": 1000,
        },
        "surface": {
            "type": "lambertian",
            "reflectance": {"type": "uniform", "value": 0.35},
        },
        "atmosphere": None,
    }
    app = RayleighSolverApp(config)
    assert app._kernel_dict.load() is not None

    # Pass a well-formed custom configuration object (with an atmosphere)
    config = {
        "mode": {
            "type": "mono",
            "wavelength": 550.
        },
        "illumination": {
            "type": "directional",
            "zenith": 0.,
            "azimuth": 0.,
            "irradiance": {"type": "uniform", "value": 1.}
        },
        "measure": {
            "type": "hemispherical",
            "zenith_res": 5.,
            "azimuth_res": 10.,
            "spp": 1000,
        },
        "surface": {
            "type": "lambertian",
            "reflectance": {"type": "uniform", "value": 0.5}
        },
        "atmosphere": {
            "type": "rayleigh_homogeneous",
            "height": 1e5,
            "sigma_s": 1e-6
        }
    }
    app = RayleighSolverApp(config)
    assert app._kernel_dict.load() is not None


@pytest.mark.slow
def test_rayleigh_solver_app_run():
    """Test the creation of a DataArray from the solver result

    We create a default scene with a set of zenith and azimuth angles,
    render the scene and create the DataArray.

    We assert the correct setting of the DataArray coordinates and dimensions,
    as well as the correct setting of data.
    """
    import numpy as np

    config = {
        "measure": {
            "type": "hemispherical",
            "zenith_res": 5.,
            "azimuth_res": 10.,
            "spp": 1000,
        }
    }

    app = RayleighSolverApp(config)
    assert eradiate.mode.type == "mono"

    app.compute()

    assert set(app.results["lo"].dims) == {"sza", "saa", "vza", "vaa", "wavelength"}

    assert app.results["lo"].attrs["angle_convention"] == "eo_scene"

    # We expect the whole [0, 360] to be covered
    assert len(app.results["lo"].coords["vaa"]) == 360 / 10 + 1
    # We expect [0, 90[ to be covered (90° should be missing)
    assert len(app.results["lo"].coords["vza"]) == 90 / 5

    # We just check that we record something as expected
    assert np.all(app.results["lo"].data > 0)


def test_rayleigh_solver_app_postprocessing():
    """Test the postprocessing method by computing the processed quantities and comparing
    them to a reference computation."""

    import numpy as np
    config = {
        "measure": {
            "type": "hemispherical",
            "zenith_res": 5.,
            "azimuth_res": 10.,
            "spp": 1000,
        },
        "illumination": {
            "type": "directional",
            "zenith": 0,
            "azimuth": 0,
            "irradiance": {"type": "uniform", "value": 5.}
        }
    }

    app = RayleighSolverApp(config)
    assert eradiate.mode.type == "mono"
    app.compute()
    app.postprocess()

    assert np.allclose(
        app.results["brdf"],
        app.results["lo"] / config["illumination"]["irradiance"]["value"]
    )
    assert np.allclose(
        app.results["brf"],
        app.results["brdf"] / np.pi
    )
