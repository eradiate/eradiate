import numpy as np

from eradiate import unit_registry as ureg
from eradiate._util import onedict_value
from eradiate.contexts import KernelDictContext
from eradiate.scenes.spectra import AirScatteringCoefficientSpectrum


def test_air_scattering_coefficient(mode_mono):
    ctx = KernelDictContext()

    # We can instantiate the class
    s = AirScatteringCoefficientSpectrum()

    # The spectrum evaluates correctly
    assert np.allclose(
        s.eval(ctx.spectral_ctx),
        ureg.Quantity(0.0114934, "km^-1"),
    )

    # The associated kernel dict is correctly formed and can be loaded
    from mitsuba.core.xml import load_dict

    assert load_dict(onedict_value(s.kernel_dict(ctx=ctx))) is not None
