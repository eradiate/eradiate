import mitsuba as mi

from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import KernelDict
from eradiate.scenes.integrators import (
    PathIntegrator,
    VolPathIntegrator,
    VolPathMISIntegrator,
)


def test_path(mode_mono):
    # Basic specification
    integrator = PathIntegrator()
    ctx = KernelDictContext()
    assert integrator.kernel_dict(ctx)["integrator"] == {"type": "path"}
    assert isinstance(
        KernelDict.from_elements(integrator, ctx=ctx).load(), mi.Integrator
    )

    # More detailed specification
    integrator = PathIntegrator(max_depth=5, rr_depth=3, hide_emitters=False)
    assert integrator.kernel_dict(ctx)["integrator"] == {
        "type": "path",
        "max_depth": 5,
        "rr_depth": 3,
        "hide_emitters": False,
    }
    assert isinstance(
        KernelDict.from_elements(integrator, ctx=ctx).load(), mi.Integrator
    )


def test_volpath(mode_mono):
    # Basic specification
    integrator = VolPathIntegrator()
    ctx = KernelDictContext()
    assert integrator.kernel_dict(ctx)["integrator"] == {"type": "volpath"}
    assert isinstance(
        KernelDict.from_elements(integrator, ctx=ctx).load(), mi.Integrator
    )

    # More detailed specification
    integrator = VolPathIntegrator(max_depth=5, rr_depth=3, hide_emitters=False)
    assert integrator.kernel_dict(ctx)["integrator"] == {
        "type": "volpath",
        "max_depth": 5,
        "rr_depth": 3,
        "hide_emitters": False,
    }
    assert isinstance(
        KernelDict.from_elements(integrator, ctx=ctx).load(), mi.Integrator
    )


def test_volpathmis(mode_mono):
    # Basic specification
    integrator = VolPathMISIntegrator()
    ctx = KernelDictContext()
    assert integrator.kernel_dict(ctx)["integrator"] == {"type": "volpathmis"}
    assert isinstance(
        KernelDict.from_elements(integrator, ctx=ctx).load(), mi.Integrator
    )

    # More detailed specification
    integrator = VolPathMISIntegrator(
        max_depth=5, rr_depth=3, hide_emitters=False, use_spectral_mis=True
    )
    assert integrator.kernel_dict(ctx)["integrator"] == {
        "type": "volpathmis",
        "max_depth": 5,
        "rr_depth": 3,
        "hide_emitters": False,
        "use_spectral_mis": True,
    }
    assert isinstance(
        KernelDict.from_elements(integrator, ctx=ctx).load(), mi.Integrator
    )
