import numpy as np
import pinttr
import pytest

from eradiate import unit_context_config as ucc
from eradiate import unit_context_kernel as uck
from eradiate import unit_registry as ureg
from eradiate._util import onedict_value
from eradiate.scenes.spectra import UniformSpectrum
from eradiate.units import PhysicalQuantity


def test_uniform(mode_mono):
    from mitsuba.core.xml import load_dict

    # Instantiate without argument
    UniformSpectrum()

    # Instantiate with only quantity
    UniformSpectrum(quantity=PhysicalQuantity.COLLISION_COEFFICIENT)
    UniformSpectrum(quantity="collision_coefficient")

    # Instantiate with unsupported quantity
    with pytest.raises(ValueError):
        UniformSpectrum(quantity="speed")

    # Instantiate with all arguments
    s = UniformSpectrum(quantity="collision_coefficient", value=1.0)
    assert s.value == ureg.Quantity(1.0, "m^-1")
    with pytest.raises(ValueError):
        UniformSpectrum(quantity="collision_coefficient", value=-1.0)

    # Raise if units and quantity are inconsistent
    with pytest.raises(pinttr.exceptions.UnitsError):
        UniformSpectrum(quantity="collision_coefficient", value=ureg.Quantity(1.0, ""))

    # Instantiate from dictionary
    s = UniformSpectrum.from_dict(
        {"quantity": "radiance", "value": 1.0, "value_units": "W/km^2/sr/nm"}
    )

    # Produced kernel dict is valid
    assert load_dict(onedict_value(s.kernel_dict())) is not None

    # Unit scaling is properly applied
    with ucc.override({"radiance": "W/m^2/sr/nm"}):
        s = UniformSpectrum(quantity="radiance", value=1.0)
    with uck.override({"radiance": "kW/m^2/sr/nm"}):
        d = s.kernel_dict()
        assert np.allclose(d["spectrum"]["value"], 1e-3)
