from eradiate.scenes.core import KernelDict
from eradiate.scenes.integrators import (
    PathIntegrator,
    VolPathIntegrator,
    VolPathMISIntegrator
)


def test_path(mode_mono):
    # Basic specification
    integrator = PathIntegrator()
    assert integrator.kernel_dict()["integrator"] == {"type": "path"}
    assert KernelDict.empty().add(integrator).load() is not None

    # More detailed specification
    integrator = PathIntegrator(max_depth=5, rr_depth=3, hide_emitters=False)
    assert integrator.kernel_dict()["integrator"] == {
        "type": "path",
        "max_depth": 5,
        "rr_depth": 3,
        "hide_emitters": False
    }
    assert KernelDict.empty().add(integrator).load() is not None


def test_volpath(mode_mono):
    # Basic specification
    integrator = VolPathIntegrator()
    assert integrator.kernel_dict()["integrator"] == {"type": "volpath"}
    assert KernelDict.empty().add(integrator).load() is not None

    # More detailed specification
    integrator = VolPathIntegrator(max_depth=5, rr_depth=3, hide_emitters=False)
    assert integrator.kernel_dict()["integrator"] == {
        "type": "volpath",
        "max_depth": 5,
        "rr_depth": 3,
        "hide_emitters": False
    }
    assert KernelDict.empty().add(integrator).load() is not None


def test_volpathmis(mode_mono):
    # Basic specification
    integrator = VolPathMISIntegrator()
    assert integrator.kernel_dict()["integrator"] == {"type": "volpathmis"}
    assert KernelDict.empty().add(integrator).load() is not None

    # More detailed specification
    integrator = VolPathMISIntegrator(max_depth=5, rr_depth=3, hide_emitters=False, use_spectral_mis=True)
    assert integrator.kernel_dict()["integrator"] == {
        "type": "volpathmis",
        "max_depth": 5,
        "rr_depth": 3,
        "hide_emitters": False,
        "use_spectral_mis": True
    }
    assert KernelDict.empty().add(integrator).load() is not None
