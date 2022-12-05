import mitsuba as mi
import pytest

from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import NodeSceneElement, traverse
from eradiate.scenes.integrators import (
    Integrator,
    PathIntegrator,
    VolPathIntegrator,
    VolPathMISIntegrator,
)
from eradiate.scenes.integrators._path_tracers import MonteCarloIntegrator
from eradiate.test_tools.types import check_type


@pytest.mark.parametrize(
    "integrator_cls",
    [PathIntegrator, VolPathIntegrator, VolPathMISIntegrator],
    ids=["path", "volpath", "volpathmis"],
)
def test_path_tracers_type(integrator_cls):
    check_type(
        integrator_cls,
        expected_mro=[MonteCarloIntegrator, Integrator, NodeSceneElement],
        expected_slots=[],
    )


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
    template, _ = traverse(integrator)
    ctx = KernelDictContext()
    kernel_dict = template.render(ctx=ctx)
    assert isinstance(mi.load_dict(kernel_dict), mi.MonteCarloIntegrator)
