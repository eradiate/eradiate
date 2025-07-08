import mitsuba as mi
import numpy as np
import pytest
import xarray as xr

import eradiate
from eradiate import unit_registry as ureg
from eradiate.experiments import DEMExperiment
from eradiate.scenes.atmosphere import HomogeneousAtmosphere
from eradiate.scenes.geometry import PlaneParallelGeometry
from eradiate.scenes.measure import MultiDistantMeasure
from eradiate.scenes.surface import DEMSurface, mesh_from_dem
from eradiate.spectral import DeltaSRF
from eradiate.test_tools.types import check_scene_element


def test_dem_experiment_construct_default(modes_all_double):
    """
    DEMExperiment initializes with default params in all modes
    """
    exp = DEMExperiment()

    # Check that the atmosphere's geometry is overridden by the experiment's
    assert exp.atmosphere.geometry is exp.geometry


def test_dem_experiment_construct_measures(modes_all):
    """
    A variety of measure specifications are acceptable
    """
    # Init with a single measure (not wrapped in a sequence)
    assert DEMExperiment(measures=MultiDistantMeasure())

    # Init from a dict-based measure spec
    # -- Correctly wrapped in a sequence
    assert DEMExperiment(measures=[{"type": "mdistant"}])

    # -- Not wrapped in a sequence
    assert DEMExperiment(measures={"type": "mdistant"})


def test_dem_experiment_construct_normalize_measures(mode_mono):
    # When setting atmosphere to None, measure target is at ground level
    exp = DEMExperiment(atmosphere=None)
    assert np.allclose(exp.measures[0].target.xyz, [0, 0, 0] * ureg.m)

    # When atmosphere is set, measure target is at ground level
    exp = DEMExperiment(atmosphere=HomogeneousAtmosphere())
    assert np.allclose(exp.measures[0].target.xyz, [0, 0, 0] * ureg.m)


def test_dem_experiment_ckd(mode_ckd, atmosphere_us_standard_ckd):
    """
    DEMExperiment with heterogeneous atmosphere in CKD mode can be created.
    """
    exp = DEMExperiment(
        atmosphere=atmosphere_us_standard_ckd,
        surface={"type": "lambertian"},
        measures={"type": "mdistant", "id": "mdistant_measure"},
    )
    check_scene_element(exp.scene, mi.Scene, ctx=exp.context_init)


def test_dem_experiment_kernel_dict(modes_all_double):
    """
    Test non-trivial kernel dict generation behaviour.
    """

    # Setting atmosphere to None
    exp = DEMExperiment(
        atmosphere=None,
        surface={"type": "lambertian"},
        measures=[
            {"type": "mdistant", "id": "mdistant_measure"},
            {"type": "radiancemeter", "origin": [1, 0, 0], "id": "radiancemeter"},
        ],
    )
    # -- Surface has default value
    mi_wrapper = check_scene_element(
        exp.scene, mi.Scene, ctx=exp.context_init, drop_parameters=False
    )  # Do not drop untracked parameters: we want to check the surface transform
    np.testing.assert_allclose(
        mi_wrapper.parameters["surface_shape.to_world"].matrix,
        mi.ScalarTransform4f().scale([5e8, 5e8, 1]).matrix,
    )
    # -- Atmosphere is not in kernel dictionary
    assert {shape.id() for shape in mi_wrapper.obj.shapes()} == {"surface_shape"}

    # -- Measures get no external medium assigned
    assert all(sensor.get_medium() is None for sensor in mi_wrapper.obj.sensors())


@pytest.mark.slow
def test_dem_experiment_real_life(mode_mono, atmosphere_us_standard_mono):
    # Construct with typical parameters
    exp = DEMExperiment(
        surface={"type": "rpv"},
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": atmosphere_us_standard_mono,
        },
        illumination={"type": "directional", "zenith": 45.0},
        measures=[
            {"type": "mdistant", "id": "mdistant_measure"},
            {"type": "radiancemeter", "origin": [1, 0, 0], "id": "radiancemeter"},
        ],
    )

    mi_wrapper = check_scene_element(exp.scene, mi.Scene, ctx=exp.context_init)

    # -- Distant measures get no external medium
    assert mi_wrapper.obj.sensors()[0].get_medium() is None

    # -- Radiancemeter inside the atmosphere must have a medium assigned
    assert mi_wrapper.obj.sensors()[1].get_medium().id() == "medium_atmosphere"


def test_dem_experiment_inconsistent_multiradiancemeter(
    mode_mono,
    atmosphere_us_standard_mono,
):
    # A MultiRadiancemeter measure must have all origins inside the atmosphere or none.
    # A mix of both will raise an error.

    # Construct with typical parameters
    exp = DEMExperiment(
        surface={"type": "rpv"},
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": atmosphere_us_standard_mono,
        },
        illumination={"type": "directional", "zenith": 45.0},
        measures=[
            {
                "type": "multi_radiancemeter",
                "origins": [[0, 0, 1], [0, 0, 1000000]],
                "directions": [[0, 0, -1], [0, 0, -1]],
                "id": "multi_radiancemeter",
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="Inconsistent placement of MultiRadiancemeterMeasure origins",
    ):
        exp.init()


def test_dem_experiment_run_basic(modes_all_double):
    """
    DEMExperiment runs successfully in all modes.
    """
    exp = DEMExperiment()
    exp.measures[0].srf = DeltaSRF(wavelengths=550.0 * ureg.nm)
    eradiate.run(exp)
    assert isinstance(exp.results, dict)


@pytest.mark.slow
def test_dem_experiment_run_detailed(modes_all_double):
    """
    Test for correctness of the result dataset generated by DEMExperiment.
    """
    # Create simple scene
    exp = DEMExperiment(
        measures=[
            {
                "type": "hemispherical_distant",
                "id": "toa_hsphere",
                "film_resolution": (32, 32),
                "spp": 1000,
                "srf": {"type": "delta", "wavelengths": 550.0 * ureg.nm},
            },
        ]
    )

    # Run RT simulation
    results = eradiate.run(exp)

    # Check result dataset structure
    # Post-processing creates expected variables ...
    expected = {"brdf", "brf", "irradiance", "radiance"}
    assert set(results.data_vars) == expected

    # ... dimensions
    assert set(results["radiance"].dims) == {"sza", "saa", "x_index", "y_index", "w"}
    assert set(results["irradiance"].dims) == {"sza", "saa", "w"}

    # ... and other coordinates
    expected_coords = {"sza", "saa", "vza", "vaa", "x", "y", "x_index", "y_index", "w"}
    if eradiate.mode().is_ckd:
        expected_coords |= {"bin_wmin", "bin_wmax"}
    assert set(results["radiance"].coords) == expected_coords

    expected_coords = {"sza", "saa", "w"}
    if eradiate.mode().is_ckd:
        expected_coords |= {"bin_wmin", "bin_wmax"}
    assert set(results["irradiance"].coords) == expected_coords

    # We just check that we record something as expected
    assert np.all(results["radiance"].data > 0.0)


def test_dem_experiment_warn_targeting_dem(modes_all_double, tmpdir):
    """
    Test that the DEMExperiment constructor raises a warning when the measure
    target is a point and a DEM is defined in the scene.
    """

    da = xr.DataArray(
        data=np.zeros((10, 10)),
        dims=["x", "y"],
        coords={
            "x": (["x"], np.linspace(-10, 20, 10), {"units": "kilometer"}),
            "y": (["y"], np.linspace(-30, 40, 10), {"units": "kilometer"}),
        },
        attrs={"units": "meter"},
    )

    mesh, xlim, ylim = mesh_from_dem(da, geometry="plane_parallel")

    with pytest.warns(UserWarning, match="uses a point target"):
        DEMExperiment(
            surface=DEMSurface.from_mesh(
                id="terrain",
                mesh=mesh,
                xlon_lim=xlim,
                ylat_lim=ylim,
                geometry=PlaneParallelGeometry(),
            ),
            measures=[
                {
                    "type": "mdistant",
                    "id": "mdistant_measure",
                    "target": {"type": "point", "xyz": [1, 1, 0]},
                },
            ],
        )
