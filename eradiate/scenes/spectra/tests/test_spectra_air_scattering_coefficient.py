import numpy as np

import eradiate
from eradiate import unit_registry as ureg
from eradiate._mode import ModeFlags
from eradiate._util import onedict_value
from eradiate.contexts import KernelDictContext
from eradiate.scenes.spectra import AirScatteringCoefficientSpectrum


def test_air_scattering_coefficient(modes_all):
    ctx = KernelDictContext()

    # We can instantiate the class
    s = AirScatteringCoefficientSpectrum()

    # The spectrum evaluates correctly (reference values computed manually)
    if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
        expected = ureg.Quantity(0.0114934, "km^-1")

    elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
        expected = ureg.Quantity(0.0114968, "km^-1")

    elif eradiate.mode().has_flags(ModeFlags.ANY_RGB):
        expected = ureg.Quantity(0.0114968, "km^-1")

    else:
        assert False

    value = s.eval(ctx.spectral_ctx)
    # assert np.allclose(value, expected)

    # The associated kernel dict is correctly formed and can be loaded
    from mitsuba.core.xml import load_dict

    # TODO: Fix this test once the issue of instantiating an RGB spectrum is solved.
    assert load_dict(s.kernel_dict(ctx=ctx)["spectrum"]) is not None
