import mitsuba as mi
import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.experiments import CanopyAtmosphereExperiment
from eradiate.scenes.atmosphere import HomogeneousAtmosphere
from eradiate.scenes.biosphere import DiscreteCanopy
from eradiate.scenes.measure import MultiDistantMeasure
from eradiate.scenes.surface import CentralPatchSurface
from eradiate.test_tools.types import check_scene_element
from eradiate.test_tools.util import skipif_data_not_found


def test_canopy_atmosphere_experiment_construct_default(mode_mono):
    """
    CanopyAtmosphereExperiment initialises with default params in all modes
    """
    exp = CanopyAtmosphereExperiment()

    # Check that the atmosphere's geometry is overridden by the experiment's
    assert exp.atmosphere.geometry is exp.geometry


def test_canopy_atmosphere_experiment_construct_measures(mode_mono):
    """
    A variety of measure specifications are acceptable
    """

    # Init with a single measure (not wrapped in a sequence)
    assert CanopyAtmosphereExperiment(measures=MultiDistantMeasure())

    # Init from a dict-based measure spec
    # -- Correctly wrapped in a sequence
    assert CanopyAtmosphereExperiment(measures=[{"type": "distant"}])

    # -- Not wrapped in a sequence
    assert CanopyAtmosphereExperiment(measures={"type": "distant"})


@pytest.mark.parametrize("padding", (0, 1))
def test_canopy_atmosphere_experiment_construct_normalize_measures(mode_mono, padding):
    # When canopy is not None, measure target matches canopy unit cell
    exp = CanopyAtmosphereExperiment(
        atmosphere=None,
        canopy=DiscreteCanopy.homogeneous(
            lai=3.0,
            leaf_radius=0.1 * ureg.m,
            l_horizontal=10.0 * ureg.m,
            l_vertical=2.0 * ureg.m,
            padding=padding,
        ),
        measures=MultiDistantMeasure(),
    )
    target = exp.measures[0].target
    canopy = exp.canopy
    assert np.isclose(target.xmin, -0.5 * canopy.size[0])
    assert np.isclose(target.xmax, 0.5 * canopy.size[0])
    assert np.isclose(target.ymin, -0.5 * canopy.size[1])
    assert np.isclose(target.ymax, 0.5 * canopy.size[1])
    assert np.isclose(target.z, canopy.size[2])

    # The measure target does not depend on the atmosphere
    exp = CanopyAtmosphereExperiment(
        geometry={"type": "plane_parallel", "width": 42.0 * ureg.km},
        atmosphere=HomogeneousAtmosphere(),
        canopy=DiscreteCanopy.homogeneous(
            lai=3.0,
            leaf_radius=0.1 * ureg.m,
            l_horizontal=10.0 * ureg.m,
            l_vertical=2.0 * ureg.m,
            padding=padding,
        ),
        measures=MultiDistantMeasure(),
    )
    target = exp.measures[0].target
    canopy = exp.canopy
    assert np.isclose(target.xmin, -0.5 * canopy.size[0])
    assert np.isclose(target.xmax, 0.5 * canopy.size[0])
    assert np.isclose(target.ymin, -0.5 * canopy.size[1])
    assert np.isclose(target.ymax, 0.5 * canopy.size[1])
    assert np.isclose(target.z, canopy.size[2])


@pytest.mark.parametrize("padding", (0, 1))
def test_canopy_atmosphere_experiment_kernel_dict(mode_mono, padding):
    # Surface width is appropriately inherited from canopy, when no atmosphere is present
    exp = CanopyAtmosphereExperiment(
        atmosphere=None,
        canopy=DiscreteCanopy.homogeneous(
            lai=3.0,
            leaf_radius=0.1 * ureg.m,
            l_horizontal=10.0 * ureg.m,
            l_vertical=2.0 * ureg.m,
            padding=padding,
        ),
        measures=[
            {"type": "distant", "id": "distant_measure"},
            {"type": "radiancemeter", "origin": [1, 0, 0], "id": "radiancemeter"},
        ],
    )

    mi_wrapper = check_scene_element(
        exp.scene, mi.Scene, drop_parameters=False
    )  # Do not drop untracked parameters: we want to check the surface transform
    assert np.allclose(
        mi_wrapper.parameters["surface_shape.to_world"].transform_affine(
            mi.Point3f(1, -1, 0)
        ),
        [5 * (2 * padding + 1), -5 * (2 * padding + 1), 0],
    )

    # -- Measures get no external medium assigned
    assert all(sensor.medium() is None for sensor in mi_wrapper.obj.sensors())

    # Surface width is appropriately inherited from atmosphere
    exp = CanopyAtmosphereExperiment(
        geometry={"type": "plane_parallel", "width": 42.0 * ureg.km},
        atmosphere=HomogeneousAtmosphere(),
        canopy=DiscreteCanopy.homogeneous(
            lai=3.0,
            leaf_radius=0.1 * ureg.m,
            l_horizontal=10.0 * ureg.m,
            l_vertical=2.0 * ureg.m,
            padding=padding,
        ),
    )

    mi_wrapper = check_scene_element(
        exp.scene, mi.Scene, drop_parameters=False
    )  # Do not drop untracked parameters: we want to check the surface transform
    assert np.allclose(
        mi_wrapper.parameters["surface_shape.to_world"].matrix,
        mi.ScalarTransform4f.scale([21000, 21000, 1]).matrix,
    )


@pytest.mark.slow
def test_canopy_atmosphere_experiment_surface_adjustment(mode_mono):
    """
    Create a CanopyAtmosphereExperiment and assert the central patch surface is created
    with the correct parameters, according to the canopy and atmosphere.
    """
    exp = CanopyAtmosphereExperiment(
        geometry={"type": "plane_parallel", "width": 42.0 * ureg.km},
        atmosphere=HomogeneousAtmosphere(),
        canopy=DiscreteCanopy.homogeneous(
            lai=3.0,
            leaf_radius=0.1 * ureg.m,
            l_horizontal=10.0 * ureg.m,
            l_vertical=2.0 * ureg.m,
        ),
        surface=CentralPatchSurface(
            bsdf={"type": "lambertian"},
            patch_bsdf={"type": "rpv"},
            patch_edges=10.0 * ureg.m,
        ),
    )

    expected = (
        mi.ScalarTransform3f.scale([1400, 1400])
        @ mi.ScalarTransform3f.translate([-0.499642857, -0.499642857])
    ).matrix

    mi_wrapper = check_scene_element(
        exp.scene, mi.Scene, drop_parameters=False
    )  # Do not drop untracked parameters: we want to check the surface texture's uv transform
    result = mi_wrapper.parameters["surface_bsdf.weight.to_uv"].matrix

    np.testing.assert_allclose(expected, result)


@pytest.mark.slow
def test_canopy_atmosphere_experiment_real_life(mode_mono):
    # Construct with typical parameters
    skipif_data_not_found(
        "spectra/absorption/us76_u86_4/us76_u86_4-spectra-18000_19000.nc"
    )
    exp = CanopyAtmosphereExperiment(
        surface={"type": "rpv"},
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": {
                "construct": "ussa_1976",
            },
        },
        canopy={
            "type": "discrete_canopy",
            "construct": "homogeneous",
            "lai": 3.0,
            "leaf_radius": 0.1 * ureg.m,
            "l_horizontal": 10.0 * ureg.m,
            "l_vertical": 2.0 * ureg.m,
        },
        illumination={"type": "directional", "zenith": 45.0},
        measures=[
            {
                "type": "distant",
                "construct": "hplane",
                "zeniths": np.arange(-60, 61, 5),
                "azimuth": 0.0,
                "id": "distant",
            },
            {"type": "radiancemeter", "origin": [1, 0, 0], "id": "radiancemeter"},
        ],
    )
    mi_wrapper = check_scene_element(exp.scene, mi.Scene, ctx=exp.context_init)

    # -- Distant measures get no external medium
    assert mi_wrapper.obj.sensors()[0].medium() is None

    # -- Radiancemeter inside the atmosphere must have a medium assigned
    assert mi_wrapper.obj.sensors()[1].medium().id() == "medium_atmosphere"


@pytest.mark.slow
def test_canopy_atmosphere_experiment_run_detailed(mode_mono):
    """
    Test for correctness of the result dataset generated by CanopyAtmosphereExperiment.
    Note: This test is outdated, most of its content should be transferred to
    tests for measure post-processing pipelines.
    """
    exp = CanopyAtmosphereExperiment(
        measures=[
            {
                "id": "toa_brf",
                "type": "distant",
                "construct": "hplane",
                "zeniths": np.arange(-60, 61, 5),
                "azimuth": 0.0,
            },
        ]
    )
    results = eradiate.run(exp)

    # Post-processing creates expected variables ...
    assert set(results.data_vars) == {
        "irradiance",
        "brf",
        "brdf",
        "radiance",
        "spp",
    }

    # ... dimensions
    assert set(results["radiance"].dims) == {"sza", "saa", "x_index", "y_index", "w"}
    assert set(results["irradiance"].dims) == {"sza", "saa", "w"}

    # ... and other coordinates
    expected_coords = {"sza", "saa", "vza", "vaa", "x", "x_index", "y", "y_index", "w"}
    assert set(results["radiance"].coords) == expected_coords
    assert set(results["irradiance"].coords) == {"sza", "saa", "w"}

    # We just check that we record something as expected
    assert np.all(results["radiance"].data > 0.0)


def test_canopy_atmosphere_experiment_inconsistent_multiradiancemeter(mode_mono):
    # A MultiRadiancemeter measure must have all origins inside the atmosphere
    # or none. A mix of both will raise an error.
    skipif_data_not_found(
        "spectra/absorption/us76_u86_4/us76_u86_4-spectra-18000_19000.nc"
    )
    # Construct with typical parameters
    exp = CanopyAtmosphereExperiment(
        surface={"type": "rpv"},
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": {
                "construct": "ussa_1976",
            },
        },
        canopy={
            "type": "discrete_canopy",
            "construct": "homogeneous",
            "lai": 3.0,
            "leaf_radius": 0.1 * ureg.m,
            "l_horizontal": 10.0 * ureg.m,
            "l_vertical": 2.0 * ureg.m,
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
