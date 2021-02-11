import numpy as np
import pytest

import eradiate
from eradiate.radprops import RadProfileFactory
from eradiate.scenes.atmosphere import HomogeneousAtmosphere
from eradiate.scenes.illumination import \
    ConstantIllumination, DirectionalIllumination
from eradiate.solvers.onedim.app import \
    OneDimScene, OneDimSolverApp, TOAHsphereMeasure, TOAPPlaneMeasure
from eradiate.util.exceptions import ModeError
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
        atmosphere=HomogeneousAtmosphere(toa_altitude=ureg.Quantity(2., "km")),
        measures=TOAPPlaneMeasure()
    )
    assert np.allclose(s.measures[0].origin, ureg.Quantity([0., 0., 2.002], "km"))


def test_onedim_solver_app_new():
    # Test the new() constructor wrapper

    # Should raise if no mode is set
    with pytest.raises(ModeError):
        eradiate.set_mode("none")
        OneDimSolverApp.new()

    # Should successfully construct a OneDimSolver otherwise
    for mode in ("mono", "mono_double"):
        eradiate.set_mode(mode)
        OneDimSolverApp.new()


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


def test_onedim_solver_app(mode_mono):
    # Test default configuration handling
    app = OneDimSolverApp()
    assert app.scene is not None
    assert app._kernel_dict is not None
    assert app._kernel_dict.load() is not None


@pytest.mark.slow
def test_onedim_solver_app_run(mode_mono):
    """Test the creation of a DataArray from the solver result

    We create a default scene with a set of zenith and azimuth angles,
    render the scene and create the DataArray.

    We assert the correct setting of the DataArray coordinates and dimensions,
    as well as the correct setting of data.
    """
    app = OneDimSolverApp(scene=OneDimScene(measures={
        "type": "toa_hsphere",
        "zenith_res": 45.,
        "azimuth_res": 180.,
        "spp": 1000,
    }))

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
    """Test the SPP splitting mechanism by running the same scene, once with
    single and double precision each.
    """
    zenith_res = 90.
    spp = 1000000
    n_split = 5

    # Double-precision run
    eradiate.set_mode("mono_double")
    app = OneDimSolverApp(scene=OneDimScene(measures=TOAPPlaneMeasure(
        zenith_res=zenith_res,
        spp=spp,
        spp_max_single=spp,
    )))
    app.run()
    results_double = app.results["toa_pplane"]["lo"].values

    # Single-precision run
    eradiate.set_mode("mono")
    app = OneDimSolverApp(scene=OneDimScene(measures=TOAPPlaneMeasure(
        zenith_res=zenith_res,
        spp=spp,
        spp_max_single=spp,
    )))
    app.run()
    results_single = app.results["toa_pplane"]["lo"].values

    # Single-precision with splitting
    app = OneDimSolverApp(scene=OneDimScene(measures=TOAPPlaneMeasure(
        zenith_res=zenith_res,
        spp=spp,
        spp_max_single=min(spp / n_split, int(1e5)),
    )))
    app.run()
    results_single_split = app.results["toa_pplane"]["lo"].values

    deviation_single = (
            np.abs(results_double - results_single) / results_double
    ).flatten()
    deviation_single_split = (
            np.abs(results_double - results_single_split) / results_double
    ).flatten()

    # Is our reference single-precision run good?
    assert np.all(deviation_single < 1e-3)

    # Is our split computation close enough to the single-precision reference?
    assert np.allclose(deviation_single, deviation_single_split, atol=5e-3)


def test_onedim_solver_app_postprocessing(mode_mono):
    """Test the postprocessing method by computing the processed quantities and
    comparing them to a reference computation.
    """
    scene = OneDimScene.from_dict({
        "measures": {
            "type": "toa_hsphere",
            "zenith_res": 5.,
            "azimuth_res": 10.,
            "spp": 1000,
            "hemisphere": "back"
        },
        "illumination": {
            "type": "directional",
            "zenith": 0,
            "azimuth": 0,
            "irradiance": {"type": "uniform", "value": 5.}
        }
    })
    app = OneDimSolverApp(scene=scene)
    app.run()

    results = app.results["toa_hsphere"]

    # Assert the correct computation of the BRDF and BRF values
    # BRDF
    brdf = ureg.Quantity(results["brdf"].values, results["brdf"].attrs["units"])
    lo = ureg.Quantity(results["lo"].values, results["lo"].attrs["units"])
    assert np.allclose(brdf, lo / scene.illumination.irradiance.value)

    # BRF
    brf = ureg.Quantity(results["brf"].values, results["brf"].attrs["units"])
    assert np.allclose(brf, brdf * np.pi)
