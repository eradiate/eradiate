import mitsuba as mi
import numpy as np

import eradiate
from eradiate import unit_context_kernel as uck
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import traverse
from eradiate.scenes.spectra import AirScatteringCoefficientSpectrum


def test_air_scattering_coefficient_construct(modes_all):
    # Construction without argument succeeds
    s = AirScatteringCoefficientSpectrum()
    assert s


def test_air_scattering_coefficient_eval(modes_all_double):
    # The spectrum evaluates correctly (reference values computed manually)
    if eradiate.mode().is_mono:
        expected = ureg.Quantity(0.0114934, "km^-1")

    elif eradiate.mode().is_ckd:
        expected = ureg.Quantity(0.0114968, "km^-1")

    else:
        raise ValueError(f"no reference value for mode {eradiate.mode()}")

    s = AirScatteringCoefficientSpectrum()
    ctx = KernelDictContext()

    value = s.eval(ctx.spectral_ctx)
    assert np.allclose(value, expected)

    # The associated kernel dict is correctly formed and can be loaded
    template, _ = traverse(s)
    ctx = KernelDictContext()
    with uck.override(length="m"):
        kernel_dict = template.render(ctx=ctx)
    assert np.isclose(kernel_dict["value"], expected.m_as("m^-1"))
    assert isinstance(mi.load_dict(kernel_dict), mi.Texture)
