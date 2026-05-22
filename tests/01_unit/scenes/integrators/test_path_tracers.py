import mitsuba as mi
import pytest

from eradiate.scenes.core import traverse
from eradiate.scenes.integrators import (
    EOVolPathIntegrator,
    PathIntegrator,
    PiecewiseVolPathIntegrator,
    VolPathIntegrator,
    VolPathMISIntegrator,
)
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "integrator_cls",
    [
        PathIntegrator,
        VolPathIntegrator,
        VolPathMISIntegrator,
        PiecewiseVolPathIntegrator,
        EOVolPathIntegrator,
    ],
    ids=["path", "volpath", "volpathmis", "piecewise_volpath", "eovolpath"],
)
@pytest.mark.parametrize(
    "kwargs",
    [{}, {"max_depth": 5, "rr_depth": 3, "hide_emitters": False}],
    ids=["noargs", "args"],
)
def test_path_tracers_construct(modes_all, integrator_cls, kwargs):
    # Construction succeed
    assert integrator_cls(**kwargs)


@pytest.mark.parametrize(
    "integrator_cls, kwargs",
    [
        (PathIntegrator, {"max_depth": 5, "rr_depth": 3, "hide_emitters": False}),
        (VolPathIntegrator, {"max_depth": 5, "rr_depth": 3, "hide_emitters": False}),
        (
            VolPathMISIntegrator,
            {
                "max_depth": 5,
                "rr_depth": 3,
                "hide_emitters": False,
                "use_spectral_mis": True,
            },
        ),
        (
            PiecewiseVolPathIntegrator,
            {
                "max_depth": 5,
                "rr_depth": 3,
                "hide_emitters": False,
            },
        ),
        (
            EOVolPathIntegrator,
            {
                "max_depth": 5,
                "rr_depth": 3,
                "rr_factor": 0.97,
                "ddis_enable": True,
                "ddis_enable_surface": True,
                "pbs_enable": True,
                "pbs_max_split_count": 50,
                "pbs_min_split_threshold": 3.0,
                "pbs_crit_rr_threshold": 0.3,
                "pbs_min_rr_threshold": 0.22,
                "nle_enable": True,
                "nle_first_clone_depth": 5,
                "nle_max_clone_depth": 12,
                "nle_nee_per_clone": 11,
                "hide_emitters": False,
            },
        ),
    ],
    ids=["path", "volpath", "volpathmis", "piecewise_volpath", "eovolpath"],
)
def test_path_tracers_kernel_dict(mode_mono, integrator_cls, kwargs):
    integrator = integrator_cls(**kwargs)
    check_scene_element(integrator, mi.MonteCarloIntegrator)


@pytest.mark.parametrize(
    "integrator_cls, kwargs",
    [
        (
            PathIntegrator,
            {"max_depth": 5, "rr_depth": 3, "hide_emitters": False, "moment": True},
        ),
        (
            VolPathIntegrator,
            {"max_depth": 5, "rr_depth": 3, "hide_emitters": False, "moment": True},
        ),
        (
            VolPathMISIntegrator,
            {
                "max_depth": 5,
                "rr_depth": 3,
                "hide_emitters": False,
                "moment": True,
            },
        ),
        (
            PiecewiseVolPathIntegrator,
            {
                "max_depth": 5,
                "rr_depth": 3,
                "hide_emitters": False,
                "moment": True,
            },
        ),
        (
            EOVolPathIntegrator,
            {
                "max_depth": 5,
                "rr_depth": 3,
                "hide_emitters": False,
                "moment": True,
            },
        ),
    ],
    ids=["path", "volpath", "volpathmis", "piecewise_volpath", "eovolpath"],
)
def test_moment_construct(mode_mono, integrator_cls, kwargs):
    integrator = integrator_cls(**kwargs)
    mi_wrapper = check_scene_element(integrator, mi.SamplingIntegrator)
    assert any("m2_nested" in name for name in mi_wrapper.obj.aov_names())


@pytest.mark.parametrize(
    "integrator_cls, kwargs",
    [
        (
            PathIntegrator,
            {"max_depth": 5, "rr_depth": 3, "hide_emitters": False, "stokes": True},
        ),
        (
            VolPathIntegrator,
            {"max_depth": 5, "rr_depth": 3, "hide_emitters": False, "stokes": True},
        ),
        (
            VolPathMISIntegrator,
            {
                "max_depth": 5,
                "rr_depth": 3,
                "hide_emitters": False,
                "stokes": True,
            },
        ),
        (
            PiecewiseVolPathIntegrator,
            {
                "max_depth": 5,
                "rr_depth": 3,
                "hide_emitters": False,
                "stokes": True,
            },
        ),
        (
            EOVolPathIntegrator,
            {
                "max_depth": 5,
                "rr_depth": 3,
                "hide_emitters": False,
                "stokes": True,
            },
        ),
    ],
    ids=["path", "volpath", "volpathmis", "piecewise_volpath", "eovolpath"],
)
def test_stokes_construct(mode_mono_polarized_single, integrator_cls, kwargs):
    integrator = integrator_cls(**kwargs)
    mi_wrapper = check_scene_element(integrator, mi.SamplingIntegrator)
    assert (
        "nested.S0.R" in mi_wrapper.obj.aov_names()
        and "nested.S1.R" in mi_wrapper.obj.aov_names()
        and "nested.S2.R" in mi_wrapper.obj.aov_names()
        and "nested.S3.R" in mi_wrapper.obj.aov_names()
    )


@pytest.mark.parametrize(
    "integrator_cls, kwargs",
    [
        (
            PathIntegrator,
            {
                "max_depth": 5,
                "rr_depth": 3,
                "hide_emitters": False,
                "stokes": True,
                "moment": True,
            },
        ),
        (
            VolPathIntegrator,
            {
                "max_depth": 5,
                "rr_depth": 3,
                "hide_emitters": False,
                "stokes": True,
                "moment": True,
            },
        ),
        (
            VolPathMISIntegrator,
            {
                "max_depth": 5,
                "rr_depth": 3,
                "hide_emitters": False,
                "stokes": True,
                "moment": True,
            },
        ),
        (
            PiecewiseVolPathIntegrator,
            {
                "max_depth": 5,
                "rr_depth": 3,
                "hide_emitters": False,
                "stokes": True,
                "moment": True,
            },
        ),
        (
            EOVolPathIntegrator,
            {
                "max_depth": 5,
                "rr_depth": 3,
                "hide_emitters": False,
                "stokes": True,
                "moment": True,
            },
        ),
    ],
    ids=["path", "volpath", "volpathmis", "piecewise_volpath", "eovolpath"],
)
def test_stokes_moment_construct(mode_mono_polarized_single, integrator_cls, kwargs):
    integrator = integrator_cls(**kwargs)
    check_scene_element(integrator, mi.SamplingIntegrator)

    kdict_template, _ = traverse(integrator)
    assert kdict_template.data["type"] == "stokes_moment"
