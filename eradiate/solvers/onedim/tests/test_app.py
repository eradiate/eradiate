from copy import deepcopy

import numpy as np
import pytest

import eradiate
from eradiate.scenes.atmosphere import HomogeneousAtmosphere
from eradiate.scenes.illumination import \
    ConstantIllumination, DirectionalIllumination
from eradiate.solvers.onedim.app import \
    OneDimScene, OneDimSolverApp, TOAHsphereMeasure, TOAPPlaneMeasure
from eradiate.util.units import ureg


def test_onedim_scene(mode_mono):
    # Construct with default parameters
    s = OneDimScene()
    assert s.kernel_dict().load() is not None

    # Test non-trivial init sequence steps

    # -- Init with a single measure (not wrapped in a sequence)
    s = OneDimScene(measures=TOAHsphereMeasure())
    assert s.kernel_dict().load() is not None
    # -- Init from a dict-based measure spec
    # ---- Correctly wrapped in a sequence
    s = OneDimScene(measures=[{"type": "toa_hsphere"}])
    assert s.kernel_dict().load() is not None
    # ---- Not wrapped in a sequence
    s = OneDimScene(measures={"type": "toa_hsphere"})
    assert s.kernel_dict().load() is not None

    # -- Check that unsupported measure-illumination configurations are rejected
    with pytest.raises(ValueError):
        OneDimScene(
            measures=TOAPPlaneMeasure(),
            illumination=ConstantIllumination()
        )

    # -- Check that surface is appropriately inherited from atmosphere
    s = OneDimScene(atmosphere=HomogeneousAtmosphere(width=ureg.Quantity(42., "km")))
    assert s.surface.width == ureg.Quantity(42., "km")

    # -- Check that principal plane measure is correctly oriented
    saa = ureg.Quantity(45., "deg")
    sza = ureg.Quantity(45., "deg")
    s = OneDimScene(
        measures=TOAPPlaneMeasure(),
        illumination=DirectionalIllumination(zenith=sza, azimuth=saa)
    )
    assert np.allclose(
        s.measures[0].orientation,
        [np.cos(saa.to(ureg.rad).m), np.sin(saa.to(ureg.rad).m), 0.]
    )

    # -- Check that TOA sensors are appropriately positioned
    s = OneDimScene(
        atmosphere=HomogeneousAtmosphere(height=ureg.Quantity(2., "km")),
        measures=TOAPPlaneMeasure()
    )
    assert np.allclose(s.measures[0].origin, ureg.Quantity([0., 0., 2.002], "km"))


@pytest.mark.slow
def test_onedim_scene_real_life(mode_mono):
    # Construct with typical parameters
    s = OneDimScene(
        surface={"type": "rpv"},
        atmosphere={"type": "heterogeneous", "profile": {"type": "us76_approx"}},
        illumination={"type": "directional", "zenith": 45.},
        measures={"type": "toa_hsphere"}
    )
    assert s.kernel_dict().load() is not None


def test_onedim_solver_app_app():
    # Test default configuration handling
    app = OneDimSolverApp()
    assert app.config == {
        "atmosphere": {"type": "homogeneous"},
        "illumination": {"type": "directional"},
        "measure": [{
            "azimuth_res": 10,
            "hemisphere": "back",
            "id": "toa_hsphere",
            "origin": [0, 0, 100.1],
            "spp": 32,
            "type": "radiancemeter_hsphere",
            "zenith_res": 10
        }],
        "mode": {"type": "mono", "wavelength": 550.0},
        "surface": {"type": "lambertian"}
    }

    # Check that the appropriate variant is selected
    assert eradiate.mode.id == "mono"

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
        "measure": [{
            "type": "toa_hsphere_lo",
            "zenith_res": 5.,
            "azimuth_res": 10.,
            "spp": 1000,
        }],
        "surface": {
            "type": "lambertian",
            "reflectance": {"type": "uniform", "value": 0.35},
        },
        "atmosphere": None,
    }
    app = OneDimSolverApp(config)
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
        "measure": [{
            "type": "toa_hsphere_lo",
            "zenith_res": 5.,
            "azimuth_res": 10.,
            "spp": 1000,
        }],
        "surface": {
            "type": "lambertian",
            "reflectance": {"type": "uniform", "value": 0.5}
        },
        "atmosphere": {
            "type": "homogeneous",
            "height": 1e5,
            "sigma_s": 1e-6
        }
    }
    app = OneDimSolverApp(config)
    assert app._kernel_dict.load() is not None

    # Test measure aliasing
    app1 = OneDimSolverApp({"measure": [{"type": "toa_hsphere", "id": "test_measure"}]})
    app2 = OneDimSolverApp({"measure": [{"type": "toa_hsphere_lo", "id": "test_measure"}]})
    app3 = OneDimSolverApp({"measure": [{"type": "toa_hsphere_brdf", "id": "test_measure"}]})
    app4 = OneDimSolverApp({"measure": [{"type": "toa_hsphere_brf", "id": "test_measure"}]})
    assert app1._kernel_dict == app2._kernel_dict
    assert app1._kernel_dict == app3._kernel_dict
    assert app1._kernel_dict == app4._kernel_dict


@pytest.mark.slow
def test_onedim_solver_app_run():
    """Test the creation of a DataArray from the solver result

    We create a default scene with a set of zenith and azimuth angles,
    render the scene and create the DataArray.

    We assert the correct setting of the DataArray coordinates and dimensions,
    as well as the correct setting of data.
    """
    import numpy as np

    config = {
        "measure": [{
            "type": "toa_hsphere",
            "zenith_res": 45.,
            "azimuth_res": 180.,
            "spp": 1000,
            "hemisphere": "back"
        }]
    }

    app = OneDimSolverApp(config)
    # Assert the correct mode of operation to be set by the application
    assert eradiate.mode.id == "mono"

    app.run()

    results = app.results["toa_hsphere"]

    # Assert the correct dimensions of the application's results
    assert set(results["lo"].dims) == {"sza", "saa", "vza", "vaa", "wavelength"}

    # We expect the whole [0, 360] to be covered
    assert len(results["lo"].coords["vaa"]) == 360. / 180.
    # # We expect [0, 90[ to be covered (90° should be missing)
    assert len(results["lo"].coords["vza"]) == 90. / 45.
    # We just check that we record something as expected
    assert np.all(results["lo"].data > 0.)


@pytest.mark.slow
def test_onedim_solver_app_spp_splitting():
    """Test the spp splitting mechanism by running the same scene, once with
    single and double precision each. The scene config contains an SPP value
    high enough for a sensor split up to occur in the single precision mode."""
    import numpy as np

    config = {
        "measure": [{
            "type": "toa_hsphere",
            "zenith_res": 45.,
            "azimuth_res": 180.,
            "spp": 1000000,
            "hemisphere": "back"
        }]
    }

    eradiate.set_mode("mono_double")
    app = OneDimSolverApp(deepcopy(config))
    app.run()
    results_double = app.results["toa_hsphere"]

    eradiate.set_mode("mono")
    app = OneDimSolverApp(deepcopy(config))
    app.run()
    results_single = app.results["toa_hsphere"]

    assert np.allclose(results_single["lo"], results_double["lo"], rtol=1e-5)


def test_rayleigh_solver_app_postprocessing():
    """Test the postprocessing method by computing the processed quantities and comparing
    them to a reference computation."""

    import numpy as np
    config = {
        "measure": [{
            "type": "toa_hsphere",
            "zenith_res": 5.,
            "azimuth_res": 10.,
            "spp": 1000,
            "hemisphere": "back"
        }],
        "illumination": {
            "type": "directional",
            "zenith": 0,
            "azimuth": 0,
            "irradiance": {"type": "uniform", "value": 5.}
        }
    }

    app = OneDimSolverApp(config)
    # Assert the correct mode of operation to be set by the application
    assert eradiate.mode.id == "mono"
    app.run()

    results = app.results["toa_hsphere"]

    # Assert the correct computation of the BRDF and BRF values
    # BRDF
    assert np.allclose(
        results["brdf"],
        results["lo"] / config["illumination"]["irradiance"]["value"]
    )
    # BRF
    assert np.allclose(
        results["brf"],
        results["brdf"] * np.pi
    )
