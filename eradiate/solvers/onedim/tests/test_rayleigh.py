import eradiate.kernel
from eradiate.scenes import SceneDict
from eradiate.solvers.onedim.rayleigh import RayleighSolverApp


def test_rayleigh_solver_app():
    # Default constructor
    app = RayleighSolverApp()
    assert app.config == SceneDict(RayleighSolverApp.DEFAULT_CONFIG)
    assert eradiate.kernel.variant() == "scalar_mono_double"

    # Check that the default scene can be instantiated
    assert app._scene_dict.load() is not None

    # custom config
    config = {
        "mode": {
            "type": "mono",
            "wavelength": 800.
        },
        "illumination": {
            "type": "directional",
            "zenith": 10.,
            "azimuth": 0.,
            "irradiance": 1.
        },
        "measure": {
            "type": "hemispherical",
            "zenith_res": 5.,
            "azimuth_res": 10.,
            "spp": 1000,
        },
        "surface": {
            "type": "lambertian",
            "reflectance": 0.35
        }
    }
    app = RayleighSolverApp(config)
    # assert app.config == config

    # custom config (with atmosphere)
    config = {
        "mode": {
            "type": "mono",
            "wavelength": 550.
        },
        "illumination": {
            "type": "directional",
            "zenith": 0.,
            "azimuth": 0.,
            "irradiance": 1.
        },
        "measure": {
            "type": "hemispherical",
            "zenith_res": 5.,
            "azimuth_res": 10.,
            "spp": 1000,
        },
        "surface": {
            "type": "lambertian",
            "reflectance": 0.5
        },
        "atmosphere": {
            "type": "rayleigh_homogeneous",
            "height": 1e5,
            "sigma_s": 1e-6
        }
    }
    app = RayleighSolverApp(config)
    # assert app.config == config

    # custom config (with custom refractive index)
    config = {
        "mode": {
            "type": "mono",
            "wavelength": 570.
        },
        "illumination": {
            "type": "directional",
            "zenith": 0.,
            "azimuth": 0.,
            "irradiance": 1.
        },
        "measure": {
            "type": "hemispherical",
            "zenith_res": 5.,
            "azimuth_res": 10.,
            "spp": 1000,
        },
        "surface": {
            "type": "lambertian",
            "reflectance": 0.5
        },
        "atmosphere": {
            "type": "rayleigh_homogeneous",
            "height": 1e5,
            "sigma_s_params": {
                "refractive_index": 1.0003
            }
        }
    }
    app = RayleighSolverApp(config)
    # assert app.config == config

    # check that wavelength from mode is included in the atmosphere config
    assert app.config["atmosphere"]["sigma_s_params"]["wavelength"] == 570.

    # check that the scattering coefficient is computed correctly
    from eradiate.scenes.atmosphere.rayleigh import sigma_s_single
    assert app._scene_dict["medium_atmosphere"]["sigma_t"]["value"] == \
           sigma_s_single(wavelength=570., refractive_index=1.0003)


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
    assert eradiate.kernel.variant() == "scalar_mono_double"

    app.compute()

    for dim in ["theta_i", "phi_i", "theta_o", "phi_o", "wavelength"]:
        assert dim in app.result.dims

    # We expect the whole [0, 360] to be covered
    assert len(app.result.coords["phi_o"]) == 360 / 10 + 1
    # We expect [0, 90[ to be covered (90° should be missing)
    assert len(app.result.coords["theta_o"]) == 90 / 5

    assert np.all(app.result.data > 0)
    assert np.allclose(
        app.result,
        app.result.sel(theta_i=0., phi_i=0., theta_o=0., phi_o=0.)
    )
