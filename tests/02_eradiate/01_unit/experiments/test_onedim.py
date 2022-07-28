import mitsuba as mi
import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.experiments import OneDimExperiment
from eradiate.experiments._onedim import measure_inside_atmosphere
from eradiate.scenes.atmosphere import (
    HeterogeneousAtmosphere,
    HomogeneousAtmosphere,
    MolecularAtmosphere,
)
from eradiate.scenes.measure import MeasureSpectralConfig, MultiDistantMeasure


def test_onedim_experiment_construct_default(modes_all_double):
    """
    OneDimExperiment initialises with default params in all modes
    """
    assert OneDimExperiment()


def test_onedim_experiment_construct_measures(modes_all):
    """
    A variety of measure specifications are acceptable
    """
    # Init with a single measure (not wrapped in a sequence)
    assert OneDimExperiment(measures=MultiDistantMeasure())

    # Init from a dict-based measure spec
    # -- Correctly wrapped in a sequence
    assert OneDimExperiment(measures=[{"type": "distant"}])
    # -- Not wrapped in a sequence
    assert OneDimExperiment(measures={"type": "distant"})


def test_onedim_experiment_construct_normalize_measures(mode_mono):
    # When setting atmosphere to None, measure target is at ground level
    exp = OneDimExperiment(atmosphere=None)
    assert np.allclose(exp.measures[0].target.xyz, [0, 0, 0] * ureg.m)

    # When atmosphere is set, measure target is at ground level
    exp = OneDimExperiment(atmosphere=HomogeneousAtmosphere(top=100.0 * ureg.km))
    assert np.allclose(exp.measures[0].target.xyz, [0, 0, 0] * ureg.m)


@pytest.mark.parametrize("bin_set", ["1nm", "10nm"])
def test_onedim_experiment_ckd(mode_ckd, bin_set):
    """
    OneDimExperiment with heterogeneous atmosphere in CKD mode can be created.
    """
    ctx = KernelDictContext(spectral_ctx={"bin_set": bin_set})
    exp = OneDimExperiment(
        atmosphere=HeterogeneousAtmosphere(
            molecular_atmosphere=MolecularAtmosphere.afgl_1986()
        ),
        surface={"type": "lambertian"},
        measures={"type": "distant", "id": "distant_measure"},
    )
    assert exp.kernel_dict(ctx=ctx).load()


def test_onedim_measure_inside_atmosphere(mode_mono):
    ctx = KernelDictContext()
    atm = HeterogeneousAtmosphere(
        geometry="plane_parallel",
        molecular_atmosphere=MolecularAtmosphere.afgl_1986(),
    )

    s1 = eradiate.scenes.measure.MultiDistantMeasure()
    assert not measure_inside_atmosphere(atm, s1, ctx)

    s2 = eradiate.scenes.measure.PerspectiveCameraMeasure(
        origin=[1, 1, 1], target=[0, 0, 0]
    )
    assert measure_inside_atmosphere(atm, s2, ctx)

    s3 = eradiate.scenes.measure.MultiRadiancemeterMeasure(
        origins=[[1, 1, 1], [0, 0, 1000000]], directions=[[0, 0, -1], [0, 0, -1]]
    )
    with pytest.raises(ValueError):
        measure_inside_atmosphere(atm, s3, ctx)


def test_onedim_experiment_kernel_dict(modes_all_double):
    """
    Test non-trivial kernel dict generation behaviour.
    """

    ctx = KernelDictContext()

    # Surface width is appropriately inherited from geometry
    exp = OneDimExperiment(
        geometry={"type": "plane_parallel", "width": 42.0, "width_units": "km"},
        atmosphere=HomogeneousAtmosphere(),
    )
    kernel_dict = exp.kernel_dict(ctx)
    assert np.allclose(
        kernel_dict["shape_surface"]["to_world"].matrix,
        mi.ScalarTransform4f.scale([21000, 21000, 1]).matrix,
    )
    assert "shape_atmosphere" in kernel_dict

    # Setting atmosphere to None
    exp = OneDimExperiment(
        geometry="plane_parallel",
        atmosphere=None,
        surface={"type": "lambertian"},
        measures=[
            {"type": "distant", "id": "distant_measure"},
            {"type": "radiancemeter", "origin": [1, 0, 0], "id": "radiancemeter"},
        ],
    )
    # -- Surface has default value
    kernel_dict = exp.kernel_dict(ctx)
    assert np.allclose(
        kernel_dict["shape_surface"]["to_world"].matrix,
        mi.ScalarTransform4f.scale([500, 500, 1]).matrix,
    )
    # -- Atmosphere is not in kernel dictionary
    assert "shape_atmosphere" not in kernel_dict

    # -- Measures get no external medium assigned
    assert "medium" not in kernel_dict["distant_measure"]
    assert "medium" not in kernel_dict["radiancemeter"]


@pytest.mark.slow
def test_onedim_experiment_real_life(mode_mono):
    ctx = KernelDictContext()

    # Construct with typical parameters
    test_absorption_data_set = eradiate.data.data_store.fetch(
        "tests/spectra/absorption/us76_u86_4-spectra-4000_25711.nc"
    )
    exp = OneDimExperiment(
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
    assert isinstance(exp.kernel_dict(ctx=ctx).load(), mi.Scene)

    # -- Distant measures get no external medium
    assert "medium" not in exp.kernel_dict(ctx=ctx)["distant_measure"]

    # -- Radiancemeter inside the atmosphere must have a medium assigned
    assert exp.kernel_dict(ctx=ctx)["radiancemeter"]["medium"] == {
        "type": "ref",
        "id": "medium_atmosphere",
    }


def test_onedim_experiment_inconsistent_multiradiancemeter(mode_mono):
    # A MultiRadiancemeter measure must have all origins inside the atmosphere or none.
    # A mix of both will raise an error.

    ctx = KernelDictContext()

    # Construct with typical parameters
    test_absorption_data_set = eradiate.data.data_store.fetch(
        "tests/spectra/absorption/us76_u86_4-spectra-4000_25711.nc"
    )
    exp = OneDimExperiment(
        geometry={"type": "plane_parallel"},
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
            {
                "type": "multi_radiancemeter",
                "origins": [[0, 0, 1], [0, 0, 1000000]],
                "directions": [[0, 0, -1], [0, 0, -1]],
                "id": "multi_radiancemeter",
            },
        ],
    )
    with pytest.raises(ValueError):
        exp.kernel_dict(ctx=ctx)


def test_onedim_experiment_run_basic(modes_all):
    """
    OneDimExperiment runs successfully in all modes.
    """
    if eradiate.mode().is_mono:
        spectral_cfg = MeasureSpectralConfig.new(wavelengths=550.0 * ureg.nm)
    elif eradiate.mode().is_ckd:
        spectral_cfg = MeasureSpectralConfig.new(bin_set="10nm", bins="550")
    else:
        pytest.skip(f"Please add test for '{eradiate.mode().id}' mode")

    exp = OneDimExperiment()
    exp.measures[0].spectral_cfg = spectral_cfg

    eradiate.run(exp)
    assert isinstance(exp.results, dict)


@pytest.mark.slow
def test_onedim_experiment_run_detailed(modes_all):
    """
    Test for correctness of the result dataset generated by OneDimExperiment.
    """
    if eradiate.mode().is_mono:
        spectral_cfg = {"wavelengths": 550.0 * ureg.nm}
    elif eradiate.mode().is_ckd:
        spectral_cfg = {"bin_set": "10nm", "bins": "550"}
    else:
        pytest.skip(f"Please add test for '{eradiate.mode().id}' mode")

    # Create simple scene
    exp = OneDimExperiment(
        measures=[
            {
                "type": "hemispherical_distant",
                "id": "toa_hsphere",
                "film_resolution": (32, 32),
                "spp": 1000,
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
