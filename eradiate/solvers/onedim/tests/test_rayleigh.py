from eradiate.solvers.onedim.rayleigh import *

def test_rayleigh_solver_app():

    # default constructor
    app = RayleighSolverApp()
    assert app.config == {
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
                "type": "distant",
                "zenith": 30.,
                "azimuth": 180.
            },
            "surface": {
                "type": "lambertian",
                "reflectance": 0.5
            }
        }

    # run with default configuration
    #assert app.run() == 0.1591796875

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
            "type": "distant",
            "zenith": 40.,
            "azimuth": 180.
        },
        "surface": {
            "type": "lambertian",
            "reflectance": 0.35
        }
    }
    app = RayleighSolverApp(config)
    assert app.config == config

    # run with custom config
    #assert app.run() == 0.1591796875

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
            "type": "distant",
            "zenith": 30.,
            "azimuth": 180.
        },
        "surface": {
            "type": "lambertian",
            "reflectance": 0.5
        },
        "atmosphere": {
            "scattering_coefficient": 1e-6,
            "height": 1e5
        }
    }
    app = RayleighSolverApp(config)
    assert app.config == config

    # run with custom config
    #assert app.run() < 0.1591796875

    # custom config (with custom refractive index)
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
            "type": "distant",
            "zenith": 30.,
            "azimuth": 180.
        },
        "surface": {
            "type": "lambertian",
            "reflectance": 0.5
        },
        "atmosphere": {
            "refractive_index": 1.0003,
            "height": 1e5
        }
    }
    app = RayleighSolverApp(config)
    assert app.config == config

    # run with custom config
    #assert app.run() < 0.1591796875

