import mitsuba as mi
import numpy as np
import pytest
import xarray as xr

import eradiate
from eradiate import unit_registry as ureg
from eradiate.constants import EARTH_RADIUS
from eradiate.experiments import AtmosphereExperiment
from eradiate.scenes.atmosphere import HomogeneousAtmosphere
from eradiate.scenes.measure import MultiDistantMeasure
from eradiate.scenes.shapes import RectangleShape
from eradiate.scenes.spectra import MultiDeltaSpectrum
from eradiate.test_tools.types import check_scene_element


def test_atmosphere_experiment_construct_default(modes_all_double):
    """
    AtmosphereExperiment initializes successfully with default parameters in
    all modes.
    """
    exp = AtmosphereExperiment()

    # Check that the atmosphere's geometry is overridden by the experiment's
    assert exp.atmosphere.geometry is exp.geometry


def test_atmosphere_experiment_extra_objects(mode_mono):
    """
    Extra objects can be added to the scene.
    """
    exp = AtmosphereExperiment(
        extra_objects={
            "reference_surface": {
                "factory": "shape",
                "type": "rectangle",
                "id": "reference_surface",
                "bsdf": {"type": "lambertian"},
            }
        }
    )
    assert isinstance(exp.extra_objects["reference_surface"], RectangleShape)
    mi_wrapper = check_scene_element(exp.scene, mi.Scene, ctx=exp.context_init)
    assert mi_wrapper.obj.shapes()[0].id() == "reference_surface"
    assert "reference_surface.bsdf.reflectance.value" in mi_wrapper.parameters.keys()


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
        ("spherical_shell", [0, 0, EARTH_RADIUS.m] * EARTH_RADIUS.u),
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
    mi_wrapper = check_scene_element(exp.scene, mi.Scene)
    # -- Surface width is inherited from geometry
    assert np.allclose(
        mi_wrapper.parameters["surface_shape.to_world"].matrix,
        mi.ScalarTransform4f.scale([21000, 21000, 1]).matrix,
    )
    # -- Atmosphere is part of the scene
    assert "shape_atmosphere" in set(shape.id() for shape in mi_wrapper.obj.shapes())
    # -- Measure gets no external medium assigned
    assert all(sensor.medium() is None for sensor in mi_wrapper.obj.sensors())

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
    mi_wrapper = check_scene_element(exp.scene, mi.Scene)
    assert np.allclose(
        mi_wrapper.parameters["surface_shape.to_world"].matrix,
        mi.ScalarTransform4f.scale([5e8, 5e8, 1]).matrix,
    )
    # -- Atmosphere is not part of the scene
    assert "shape_atmosphere" not in set(
        shape.id() for shape in mi_wrapper.obj.shapes()
    )
    # -- Measures get no external medium assigned
    assert all(sensor.medium() is None for sensor in mi_wrapper.obj.sensors())


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
                "absorption_dataset": test_absorption_data_set,
            },
        },
        illumination={"type": "directional", "zenith": 45.0},
        measures=[
            {"type": "distant", "id": "distant_measure"},
            {"type": "radiancemeter", "origin": [1, 0, 0], "id": "radiancemeter"},
        ],
    )
    mi_wrapper = check_scene_element(exp.scene, mi.Scene, ctx=exp.context_init)

    # -- Distant measures get no external medium
    mi_sensors = {sensor.id(): sensor for sensor in mi_wrapper.obj.sensors()}
    assert mi_sensors["distant_measure"].medium() is None

    # -- Radiancemeter inside the atmosphere must have a medium assigned
    assert mi_sensors["radiancemeter"].medium().id() == "medium_atmosphere"


@pytest.mark.slow
def test_atmosphere_experiment_run_basic(modes_all_double):
    if eradiate.mode().is_mono:
        molecular_atmosphere_params = {
            "construct": "ussa_1976",
            "absorption_dataset": "tests/spectra/absorption/us76_u86_4-spectra-4000_25711.nc",
        }
    else:
        molecular_atmosphere_params = {
            "construct": "afgl_1986",
            # "absorption_dataset": "ckd/absorption/10nm/afgl_1986-us_standard-10nm.nc"
        }

    exp = AtmosphereExperiment(
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": {
                "type": "molecular",
                **molecular_atmosphere_params,
            },
            "particle_layers": [{"type": "particle_layer"}],
        },
        surface={"type": "lambertian"},
        measures={
            "type": "distant",
            "id": "distant_measure",
            "srf": MultiDeltaSpectrum(wavelengths=550.0 * ureg.nm),
        },
    )
    assert isinstance(eradiate.run(exp, spp=100), xr.Dataset)


@pytest.mark.slow
def test_atmosphere_experiment_run_detailed(modes_all):
    """
    Test for correctness of the result dataset generated by AtmosphereExperiment.
    """
    # Create simple scene
    exp = AtmosphereExperiment(
        measures=[
            {
                "type": "hemispherical_distant",
                "id": "toa_hsphere",
                "film_resolution": (32, 32),
                "spp": 100,
                "srf": MultiDeltaSpectrum(wavelengths=550.0 * ureg.nm),
            },
        ]
    )

    # Run RT simulation
    results = eradiate.run(exp)

    # Check result dataset structure
    # Post-processing creates expected variables ...
    expected = {"irradiance", "brf", "brdf", "radiance", "spp"}
    if eradiate.mode().is_ckd:
        expected |= {"irradiance", "brf", "brdf", "radiance"}
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
