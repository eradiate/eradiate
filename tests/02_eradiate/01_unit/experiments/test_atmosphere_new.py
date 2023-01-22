import mitsuba as mi
import numpy as np
import pytest
import xarray as xr

import eradiate
from eradiate import unit_registry as ureg
from eradiate.experiments import AtmosphereExperiment
from eradiate.scenes.atmosphere import HomogeneousAtmosphere
from eradiate.scenes.measure import MultiDistantMeasure
from eradiate.test_tools.types import check_scene_element


def test_atmosphere_experiment_construct_default(modes_all_double):
    """
    AtmosphereExperiment initialises successfully with default parameters in
    all modes.
    """
    assert AtmosphereExperiment()


def test_atmosphere_experiment_construct_measures(modes_all_double):
    """
    A variety of measure specifications are acceptable.
    """
    # Init with a single measure (not wrapped in a sequence)
    assert AtmosphereExperiment(measures=MultiDistantMeasure())

    # Init from a dict-based measure spec
    # -- Correctly wrapped in a sequence
    assert AtmosphereExperiment(measures=[{"type": "distant"}])
    # -- Not wrapped in a sequence
    assert AtmosphereExperiment(measures={"type": "distant"})


@pytest.mark.parametrize(
    "geometry, expected",
    [
        ("plane_parallel", [0, 0, 0] * ureg.m),
        ("spherical_shell", [0, 0, 6378.1] * ureg.km),
    ],
    ids=["plane_parallel", "spherical_shell"],
)
def test_atmosphere_experiment_construct_normalize_measures(
    mode_mono, geometry, expected
):
    """
    Default measure target is set to ground level.
    """
    exp = AtmosphereExperiment(geometry=geometry)
    assert np.allclose(exp.measures[0].target.xyz, expected)


def test_atmosphere_experiment_kernel_dict(mode_mono):
    """
    Test non-trivial kernel dict generation behaviour.
    """

    # With an atmosphere
    exp = AtmosphereExperiment(
        geometry={"type": "plane_parallel", "width": 42.0, "width_units": "km"},
        atmosphere=HomogeneousAtmosphere(),
        measures={"type": "distant"},
    )
    mi_scene, mi_params = check_scene_element(exp.scene, mi.Scene)
    # -- Surface width is inherited from geometry
    assert np.allclose(
        mi_params["surface_shape.to_world"].matrix,
        mi.ScalarTransform4f.scale([21000, 21000, 1]).matrix,
    )
    # -- Atmosphere is part of the scene
    assert "shape_atmosphere" in set(shape.id() for shape in mi_scene.shapes())
    # -- Measure gets no external medium assigned
    assert all(sensor.medium() is None for sensor in mi_scene.sensors())

    # Without an atmosphere
    exp = AtmosphereExperiment(
        geometry="plane_parallel",
        atmosphere=None,
        surface={"type": "lambertian"},
        measures=[
            {"type": "distant", "id": "distant_measure"},
            {"type": "radiancemeter", "origin": [1, 0, 0], "id": "radiancemeter"},
        ],
    )
    # -- Surface width has default value
    mi_scene, mi_params = check_scene_element(exp.scene, mi.Scene)
    assert np.allclose(
        mi_params["surface_shape.to_world"].matrix,
        mi.ScalarTransform4f.scale([5e8, 5e8, 1]).matrix,
    )
    # -- Atmosphere is not part of the scene
    assert "shape_atmosphere" not in set(shape.id() for shape in mi_scene.shapes())
    # -- Measures get no external medium assigned
    assert all(sensor.medium() is None for sensor in mi_scene.sensors())


@pytest.mark.slow
def test_atmosphere_experiment_real_life(mode_mono):
    # Construct with typical parameters
    test_absorption_data_set = eradiate.data.data_store.fetch(
        "tests/spectra/absorption/us76_u86_4-spectra-4000_25711.nc"
    )
    exp = AtmosphereExperiment(
        surface={"type": "rpv"},
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": {
                "construct": "ussa_1976",
                "absorption_data_sets": dict(us76_u86_4=test_absorption_data_set),
            },
        },
        illumination={"type": "directional", "zenith": 45.0},
        measures=[
            {"type": "distant", "id": "distant_measure"},
            {"type": "radiancemeter", "origin": [1, 0, 0], "id": "radiancemeter"},
        ],
    )
    mi_scene, mi_params = check_scene_element(exp.scene, mi.Scene)

    # -- Distant measures get no external medium
    mi_sensors = {sensor.id(): sensor for sensor in mi_scene.sensors()}
    assert mi_sensors["distant_measure"].medium() is None
    # -- Radiancemeter inside the atmosphere must have a medium assigned
    assert mi_sensors["radiancemeter"].medium() is not None


def test_atmosphere_experiment_run_basic(modes_all_double):
    exp = AtmosphereExperiment(
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": {
                "type": "molecular",
                "construct": "afgl_1986" if eradiate.mode().is_ckd else "ussa_1976",
            },
            "particle_layers": [{"type": "particle_layer"}],
        },
        surface={"type": "lambertian"},
        measures={
            "type": "distant",
            "id": "distant_measure",
            "spectral_cfg": {"bins": ["550"]}
            if eradiate.mode().is_ckd
            else {"wavelengths": [550.0] * ureg.nm},
        },
    )
    assert isinstance(eradiate.run(exp, spp=100), xr.Dataset)


@pytest.mark.slow
def test_onedim_experiment_run_detailed(modes_all):
    """
    Test for correctness of the result dataset generated by AtmosphereExperiment.
    """
    if eradiate.mode().is_mono:
        spectral_cfg = {"wavelengths": 550.0 * ureg.nm}
    elif eradiate.mode().is_ckd:
        spectral_cfg = {"bin_set": "10nm", "bins": "550"}
    else:
        pytest.skip(f"Please add test for '{eradiate.mode().id}' mode")

    # Create simple scene
    exp = AtmosphereExperiment(
        measures=[
            {
                "type": "hemispherical_distant",
                "id": "toa_hsphere",
                "film_resolution": (32, 32),
                "spp": 100,
                "spectral_cfg": spectral_cfg,
            },
        ]
    )

    # Run RT simulation
    results = eradiate.run(exp)

    # Check result dataset structure
    # Post-processing creates expected variables ...
    expected = {"irradiance", "brf", "brdf", "radiance", "spp", "srf"}
    if eradiate.mode().is_ckd:
        expected |= {"irradiance_srf", "brf_srf", "brdf_srf", "radiance_srf"}
    assert set(results.data_vars) == expected

    # ... dimensions
    assert set(results["radiance"].dims) == {"sza", "saa", "x_index", "y_index", "w"}
    assert set(results["irradiance"].dims) == {"sza", "saa", "w"}

    # ... and other coordinates
    expected_coords = {"sza", "saa", "vza", "vaa", "x", "y", "x_index", "y_index", "w"}
    if eradiate.mode().is_ckd:
        expected_coords |= {"bin", "bin_wmin", "bin_wmax"}
    assert set(results["radiance"].coords) == expected_coords

    expected_coords = {"sza", "saa", "w"}
    if eradiate.mode().is_ckd:
        expected_coords |= {"bin", "bin_wmin", "bin_wmax"}
    assert set(results["irradiance"].coords) == expected_coords

    # We just check that we record something as expected
    assert np.all(results["radiance"].data > 0.0)
