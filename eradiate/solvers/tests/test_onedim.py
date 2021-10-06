import numpy as np
import pytest

import eradiate
from eradiate import path_resolver
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.exceptions import ModeError
from eradiate.scenes.atmosphere import (
    HeterogeneousAtmosphere,
    HomogeneousAtmosphere,
    MolecularAtmosphere,
)
from eradiate.scenes.measure._distant import DistantRadianceMeasure
from eradiate.solvers.onedim import OneDimScene, OneDimSolverApp


def test_onedim_scene(modes_all_mono_ckd):
    from mitsuba.core import ScalarTransform4f

    ctx = KernelDictContext()

    # Construct with default parameters
    s = OneDimScene()
    assert s.kernel_dict(ctx=ctx).load() is not None

    # Test non-trivial init sequence steps

    # -- Init with a single measure (not wrapped in a sequence)
    s = OneDimScene(measures=DistantRadianceMeasure())
    assert s.kernel_dict(ctx=ctx).load() is not None
    # -- Init from a dict-based measure spec
    # ---- Correctly wrapped in a sequence
    s = OneDimScene(measures=[{"type": "distant_radiance"}])
    assert s.kernel_dict(ctx=ctx).load() is not None
    # ---- Not wrapped in a sequence
    s = OneDimScene(measures={"type": "distant_radiance"})
    assert s.kernel_dict(ctx=ctx).load() is not None

    # -- Surface width is appropriately inherited from atmosphere
    s = OneDimScene(atmosphere=HomogeneousAtmosphere(width=ureg.Quantity(42.0, "km")))
    kernel_dict = s.kernel_dict(ctx)
    assert np.allclose(
        kernel_dict["surface"]["to_world"].matrix,
        ScalarTransform4f.scale([21000, 21000, 1]).matrix,
    )

    # -- Setting atmosphere to None
    s = OneDimScene(
        atmosphere=None,
        surface={"type": "lambertian", "width": 100.0, "width_units": "m"},
        measures={"type": "distant_radiance", "id": "distant_measure"},
    )
    # -- Surface width is not overridden
    kernel_dict = s.kernel_dict(ctx)
    assert np.allclose(
        kernel_dict["surface"]["to_world"].matrix,
        ScalarTransform4f.scale([50, 50, 1]).matrix,
    )
    # -- Measure target is at ground level
    assert np.allclose(s.measures[0].target.xyz, [0, 0, 0] * ureg.m)
    # -- Measure ray origins are projected to a sphere of radius 1 m
    assert np.allclose(s.measures[0].origin.radius, 1.0 * ureg.m)
    # -- Atmosphere is not in kernel dictionary
    assert "atmosphere" not in kernel_dict


def test_onedim_scene_ckd(mode_ckd):
    """
    OneDimScene with heterogeneous atmosphere in CKD mode can be created.
    """
    ctx = KernelDictContext()
    s = OneDimScene(
        atmosphere=HeterogeneousAtmosphere(
            molecular_atmosphere=MolecularAtmosphere.afgl1986()
        ),
        surface={"type": "lambertian"},
        measures={"type": "distant_radiance", "id": "distant_measure"},
    )
    assert s.kernel_dict(ctx=ctx).load() is not None


def test_onedim_solver_app_new():
    # Test the new() constructor wrapper

    # Should raise if no mode is set
    with pytest.raises(ModeError):
        eradiate.set_mode("none")
        with pytest.warns(DeprecationWarning):
            OneDimSolverApp.new()

    # Should successfully construct a OneDimSolver otherwise
    for mode in ("mono", "mono_double", "ckd", "ckd_double"):
        eradiate.set_mode(mode)
        with pytest.warns(DeprecationWarning):
            OneDimSolverApp.new()


@pytest.mark.slow
def test_onedim_scene_real_life(mode_mono):
    ctx = KernelDictContext()

    # Construct with typical parameters
    test_absorption_data_set = path_resolver.resolve(
        "tests/spectra/absorption/us76_u86_4-spectra-4000_25711.nc"
    )
    s = OneDimScene(
        surface={"type": "rpv"},
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": {
                "construct": "ussa1976",
                "absorption_data_sets": dict(us76_u86_4=test_absorption_data_set),
            },
        },
        illumination={"type": "directional", "zenith": 45.0},
        measures={"type": "distant_reflectance", "id": "toa"},
    )
    assert s.kernel_dict(ctx=ctx).load() is not None


def test_onedim_solver_app_construct(modes_all_mono_ckd):
    # Test default configuration handling
    with pytest.warns(DeprecationWarning):
        app = OneDimSolverApp()
    assert app.scene is not None


@pytest.mark.slow
def test_onedim_solver_app_run(mode_mono):
    """
    Test the creation of a DataArray from the solver result

    We create a default scene with a set of zenith and azimuth angles,
    render the scene and create the DataArray.

    We assert the correct setting of the DataArray coordinates and dimensions,
    as well as the correct setting of data.
    """
    with pytest.warns(DeprecationWarning):
        app = OneDimSolverApp(
            scene=OneDimScene(
                measures=[
                    {
                        "type": "distant_reflectance",
                        "id": "toa_hsphere",
                        "film_resolution": (32, 32),
                        "spp": 1000,
                    },
                ]
            )
        )

    app.run()

    results = app.results["toa_hsphere"]

    # Post-processing creates expected variables ...
    assert set(results.data_vars) == {"irradiance", "brf", "brdf", "lo", "spp"}
    # ... dimensions
    assert set(results["lo"].dims) == {"sza", "saa", "x", "y", "w", "channel"}
    assert set(results["irradiance"].dims) == {"sza", "saa", "w"}
    # ... and other coordinates
    assert set(results["lo"].coords) == {
        "sza",
        "saa",
        "vza",
        "vaa",
        "x",
        "y",
        "w",
        "channel",
    }
    assert set(results["irradiance"].coords) == {"sza", "saa", "w"}

    # We just check that we record something as expected
    assert np.all(results["lo"].data > 0.0)


def test_onedim_solver_app_postprocessing(mode_mono):
    """
    Test the postprocessing method by computing the processed quantities and
    comparing them to a reference computation.
    """
    scene = OneDimScene(
        measures={
            "type": "distant_reflectance",
            "id": "toa_hsphere",
            "film_resolution": (32, 32),
            "spp": 1000,
        },
        illumination={
            "type": "directional",
            "zenith": 0.0,
            "azimuth": 0.0,
            "irradiance": {"type": "uniform", "value": 5.0},
        },
    )

    with pytest.warns(DeprecationWarning):
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
