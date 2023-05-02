import mitsuba as mi
import numpy as np

import eradiate
from eradiate import KernelContext
from eradiate import unit_context_kernel as uck
from eradiate import unit_registry as ureg
from eradiate.scenes.spectra import AirScatteringCoefficientSpectrum
from eradiate.test_tools.types import check_scene_element


def test_air_scattering_coefficient_construct(modes_all):
    # Construction without argument succeeds
    s = AirScatteringCoefficientSpectrum()
    assert s


def test_air_scattering_coefficient_eval(modes_all_double):
    # The spectrum evaluates correctly (reference values computed manually)
    expected = 0.0114934 / ureg.km  # value at 550 nm (default wavelength)

    s = AirScatteringCoefficientSpectrum()
    ctx = KernelContext()

    value = s.eval(ctx.si)
    assert np.allclose(value, expected)


def test_air_scattering_coefficient_kernel_dict(modes_all):
    s = AirScatteringCoefficientSpectrum()

    with uck.override(length="m"):
        mi_wrapper = check_scene_element(s, mi.Texture)

    if eradiate.mode().is_mono:
        expected = ureg.convert(0.0114934, "km^-1", "m^-1")

    elif eradiate.mode().is_ckd:
        expected = ureg.convert(0.0114968, "km^-1", "m^-1")

    else:
        raise ValueError(f"no reference value for mode {eradiate.mode()}")

    assert np.isclose(mi_wrapper.parameters["value"], expected)
