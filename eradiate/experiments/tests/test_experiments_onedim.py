import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate._mode import ModeFlags
from eradiate.contexts import KernelDictContext
from eradiate.exceptions import ModeError, UnsupportedModeError
from eradiate.experiments._onedim import OneDimExperiment
from eradiate.scenes.atmosphere import (
    HeterogeneousAtmosphere,
    HomogeneousAtmosphere,
    MolecularAtmosphere,
)
from eradiate.scenes.measure import DistantRadianceMeasure, MeasureSpectralConfig


def test_onedim_experiment_construct_default(modes_all):
    """
    OneDimExperiment initialises with default params in all modes
    """
    assert OneDimExperiment()


def test_onedim_experiment_construct_measures(modes_all):
    """
    A variety of measure specifications are acceptable
    """
    # Init with a single measure (not wrapped in a sequence)
    assert OneDimExperiment(measures=DistantRadianceMeasure())

    # Init from a dict-based measure spec
    # -- Correctly wrapped in a sequence
    assert OneDimExperiment(measures=[{"type": "distant_radiance"}])
    # -- Not wrapped in a sequence
    assert OneDimExperiment(measures={"type": "distant_radiance"})


def test_onedim_experiment_construct_normalize_measures(mode_mono):
    # When setting atmosphere to None, measure target is at ground level
    exp = OneDimExperiment(atmosphere=None)
    assert np.allclose(exp.measures[0].target.xyz, [0, 0, 0] * ureg.m)
    # ... and measure ray origins are projected to a sphere of radius 1 m
    assert np.allclose(exp.measures[0].origin.radius, 1.0 * ureg.m)

    # When atmosphere is set, measure target is at TOA
    exp = OneDimExperiment(atmosphere=HomogeneousAtmosphere(top=100.0 * ureg.km))
    assert np.allclose(
        exp.measures[0].target.xyz, [0, 0, exp.atmosphere.top.m_as(ureg.m)] * ureg.m
    )
    # ... and measure ray origins are projected to a sphere of radius TOA / 100
    assert np.allclose(exp.measures[0].origin.radius, 0.01 * exp.atmosphere.top)


def test_onedim_experiment_ckd(mode_ckd):
    """
    OneDimExperiment with heterogeneous atmosphere in CKD mode can be created.
    """
    ctx = KernelDictContext()
    exp = OneDimExperiment(
        atmosphere=HeterogeneousAtmosphere(
            molecular_atmosphere=MolecularAtmosphere.afgl1986()
        ),
        surface={"type": "lambertian"},
        measures={"type": "distant_radiance", "id": "distant_measure"},
    )
    assert exp.kernel_dict(ctx=ctx).load() is not None


def test_onedim_experiment_kernel_dict(modes_all):
    """
    Test non-trivial kernel dict generation behaviour.
    """
    from mitsuba.core import ScalarTransform4f

    ctx = KernelDictContext()

    # Surface width is appropriately inherited from atmosphere
    exp = OneDimExperiment(
        atmosphere=HomogeneousAtmosphere(width=ureg.Quantity(42.0, "km"))
    )
    kernel_dict = exp.kernel_dict(ctx)
    assert np.allclose(
        kernel_dict["surface"]["to_world"].matrix,
        ScalarTransform4f.scale([21000, 21000, 1]).matrix,
    )

    # Setting atmosphere to None
    exp = OneDimExperiment(
        atmosphere=None,
        surface={"type": "lambertian", "width": 100.0, "width_units": "m"},
        measures={"type": "distant_radiance", "id": "distant_measure"},
    )
    # -- Surface width is not overridden
    kernel_dict = exp.kernel_dict(ctx)
    assert np.allclose(
        kernel_dict["surface"]["to_world"].matrix,
        ScalarTransform4f.scale([50, 50, 1]).matrix,
    )
    # -- Atmosphere is not in kernel dictionary
    assert "atmosphere" not in kernel_dict


@pytest.mark.slow
def test_onedim_experiment_real_life(mode_mono):
    ctx = KernelDictContext()

    # Construct with typical parameters
    test_absorption_data_set = eradiate.path_resolver.resolve(
        "tests/spectra/absorption/us76_u86_4-spectra-4000_25711.nc"
    )
    exp = OneDimExperiment(
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
    assert exp.kernel_dict(ctx=ctx).load() is not None


def test_onedim_experiment_run_basic(modes_all_mono_ckd):
    """
    OneDimExperiment runs successfully in all modes.
    """
    if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
        spectral_cfg = MeasureSpectralConfig.new(wavelengths=550.0 * ureg.nm)
    elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
        spectral_cfg = MeasureSpectralConfig.new(bin_set="10nm_test", bins="550")
    else:
        raise ModeError("test is yet to be defined")

    exp = OneDimExperiment()
    exp.measures[0].spectral_cfg = spectral_cfg

    exp.run()
    assert isinstance(exp.results, dict)


@pytest.mark.slow
def test_onedim_experiment_run_detailed(modes_all_mono_ckd):
    """
    Test for correctness of the result dataset generated by OneDimExperiment.
    Note: This test is outdated, most of its content should be transferred to
    tests for measure post-processing pipelines.
    """
    if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
        spectral_cfg = {"wavelengths": 550.0 * ureg.nm}
    elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
        spectral_cfg = {"bin_set": "10nm", "bins": "550"}
    else:
        raise UnsupportedModeError("Test is yet to be written")

    # Create simple scene
    exp = OneDimExperiment(
        measures=[
            {
                "type": "distant_reflectance",
                "id": "toa_hsphere",
                "film_resolution": (32, 32),
                "spp": 1000,
                "spectral_cfg": spectral_cfg,
            },
        ]
    )

    # Run RT simulation
    exp.run()

    # Check result dataset structure
    results = exp.results["toa_hsphere"]

    # Post-processing creates expected variables ...
    expected = {"irradiance", "brf", "brdf", "lo"}

    if not eradiate.mode().has_flags(ModeFlags.ANY_CKD):
        expected.add("spp")  # This variable is irrelevant in non-CKD modes

    assert set(results.data_vars) == expected

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
