import mitsuba as mi
import pytest

from eradiate.scenes.integrators import (
    PathIntegrator,
    VolPathIntegrator,
    VolPathMISIntegrator,
)
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "integrator_cls",
    [PathIntegrator, VolPathIntegrator, VolPathMISIntegrator],
    ids=["path", "volpath", "volpathmis"],
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
    ],
    ids=["path", "volpath", "volpathmis"],
)
def test_path_tracers_kernel_dict(mode_mono, integrator_cls, kwargs):
    integrator = integrator_cls(**kwargs)
    check_scene_element(integrator, mi.MonteCarloIntegrator)
