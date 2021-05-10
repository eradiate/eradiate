import pinttr
import pytest

from eradiate import unit_registry as ureg
from eradiate._util import onedict_value
from eradiate.contexts import KernelDictContext, SpectralContext
from eradiate.scenes.core import KernelDict
from eradiate.scenes.illumination import DirectionalIllumination


def test_directional(mode_mono):
    from mitsuba.core.xml import load_dict

    # We need a default spectral config
    ctx = KernelDictContext()

    # Constructor
    d = DirectionalIllumination()
    assert load_dict(onedict_value(d.kernel_dict(ctx))) is not None

    # Check if a more detailed spec is valid
    d = DirectionalIllumination(irradiance={"type": "uniform", "value": 1.0})
    assert load_dict(onedict_value(d.kernel_dict(ctx))) is not None

    # Check if solar irradiance spectrum can be used
    d = DirectionalIllumination(irradiance={"type": "solar_irradiance"})
    assert load_dict(onedict_value(d.kernel_dict(ctx))) is not None

    # Check if specification from a float works
    d = DirectionalIllumination(irradiance=1.0)
    assert load_dict(onedict_value(d.kernel_dict(ctx))) is not None

    # Check if specification from a constant works
    d = DirectionalIllumination(irradiance=ureg.Quantity(1.0, "W/m^2/nm"))
    assert load_dict(onedict_value(d.kernel_dict(ctx))) is not None
    with pytest.raises(pinttr.exceptions.UnitsError):  # Wrong units
        DirectionalIllumination(irradiance=ureg.Quantity(1.0, "W/m^2/sr/nm"))
