import mitsuba as mi
import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.exceptions import UnsupportedModeError
from eradiate.experiments import CanopyExperiment
from eradiate.scenes.biosphere import DiscreteCanopy
from eradiate.scenes.measure import MultiDistantMeasure
from eradiate.scenes.spectra import MultiDeltaSpectrum
from eradiate.test_tools.types import check_scene_element


def test_canopy_experiment_construct_default(modes_all_double):
    """
    CanopyExperiment initializes with default params in all modes
    """
    assert CanopyExperiment()


def test_canopy_experiment_construct_measures(mode_mono_double):
    """
    A variety of measure specifications are acceptable
    """

    # Init with a single measure (not wrapped in a sequence)
    assert CanopyExperiment(measures=MultiDistantMeasure())

    # Init from a dict-based measure spec
    # -- Correctly wrapped in a sequence
    assert CanopyExperiment(measures=[{"type": "distant"}])
    # -- Not wrapped in a sequence
    assert CanopyExperiment(measures={"type": "distant"})


@pytest.mark.parametrize("padding", (0, 1))
def test_canopy_experiment_construct_normalize_measures(mode_mono_double, padding):
    """
    When canopy is not None, measure target matches canopy unit cell
    """
    exp = CanopyExperiment(
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


@pytest.mark.parametrize("padding", (0, 1))
def test_canopy_experiment_kernel_dict(modes_all_double, padding):
    # Surface width is appropriately inherited from canopy
    exp = CanopyExperiment(
        canopy=DiscreteCanopy.homogeneous(
            lai=0.5,  # This very low value ensures fast object initialization
            leaf_radius=0.1 * ureg.m,
            l_horizontal=10.0 * ureg.m,
            l_vertical=2.0 * ureg.m,
            padding=padding,
        )
    )
    mi_wrapper = check_scene_element(
        exp.scene, mi.Scene, drop_parameters=False
    )  # Don't drop untracked params at this stage: we want to check the surface transform

    assert np.allclose(
        mi_wrapper.parameters["surface_shape.to_world"].transform_affine(
            mi.Point3f(1, -1, 0)
        ),
        [5 * (2 * padding + 1), -5 * (2 * padding + 1), 0],
    )


@pytest.mark.slow
def test_canopy_experiment_real_life(modes_all_double):
    # Construct with typical parameters
    exp = CanopyExperiment(
        surface={"type": "lambertian"},
        canopy={
            "type": "discrete_canopy",
            "construct": "homogeneous",
            "lai": 3.0,
            "leaf_radius": 0.1 * ureg.m,
            "l_horizontal": 10.0 * ureg.m,
            "l_vertical": 2.0 * ureg.m,
        },
        illumination={"type": "directional", "zenith": 45.0},
        measures={
            "type": "distant",
            "construct": "hplane",
            "zeniths": np.arange(-60, 61, 5),
            "azimuth": 0.0,
        },
    )
    check_scene_element(exp.scene, mi.Scene)


@pytest.mark.slow
def test_canopy_experiment_run_detailed(modes_all_double):
    """
    Test for correctness of the result dataset generated by CanopyExperiment.
    Note: This test is outdated, most of its content should be transferred to
    tests for measure post-processing pipelines.
    """
    srf = MultiDeltaSpectrum(wavelengths=550.0 * ureg.nm)
    if eradiate.mode().is_mono:
        expected_vars = {
            "irradiance",
            "brf",
            "brdf",
            "radiance",
            "spp",
        }
        expected_coords_radiance = {
            "sza",
            "saa",
            "vza",
            "vaa",
            "x_index",
            "x",
            "y_index",
            "y",
            "w",
        }
        expected_coords_irradiance = {"sza", "saa", "w"}

    elif eradiate.mode().is_ckd:
        expected_vars = {
            "irradiance",
            "brf",
            "brdf",
            "radiance",
            "spp",
            "wbounds",
        }
        expected_coords_radiance = {
            "sza",
            "saa",
            "vza",
            "vaa",
            "x_index",
            "x",
            "y_index",
            "y",
            "w",
        }
        expected_coords_irradiance = {
            "sza",
            "saa",
            "w",
        }

    else:
        raise UnsupportedModeError

    exp = CanopyExperiment(
        measures=[
            {
                "type": "hemispherical_distant",
                "id": "toa_hsphere",
                "film_resolution": (32, 32),
                "spp": 1000,
                "srf": srf,
            },
        ]
    )

    results = eradiate.run(exp)

    # Post-processing creates expected variables ...
    assert set(results.data_vars) == expected_vars

    # ... dimensions
    assert set(results["radiance"].dims) == {"sza", "saa", "x_index", "y_index", "w"}
    assert set(results["irradiance"].dims) == {"sza", "saa", "w"}

    # ... and other coordinates
    assert set(results["radiance"].coords) == expected_coords_radiance
    assert set(results["irradiance"].coords) == expected_coords_irradiance

    # We just check that we record something as expected
    assert np.all(results["radiance"].data > 0.0)
