import numpy as np
import pinttr
import pytest

import eradiate
from eradiate.scenes.spectra import (
    SolarIrradianceSpectrum,
    SpectrumFactory,
    UniformSpectrum,
)
from eradiate._util import onedict_value
from eradiate._units import PhysicalQuantity
from eradiate import unit_registry as ureg
from eradiate import unit_context_config as ucc
from eradiate import unit_context_kernel as uck


def test_converter(mode_mono):
    # Dicts are correctly processed
    s = SpectrumFactory.converter("radiance")({"type": "uniform"})
    assert s == UniformSpectrum(quantity="radiance", value=1.0)
    s = SpectrumFactory.converter("irradiance")({"type": "uniform"})
    assert s == UniformSpectrum(quantity="irradiance", value=1.0)

    # Floats and quantities are correctly processed
    s = SpectrumFactory.converter("radiance")(1.0)
    assert s == UniformSpectrum(quantity="radiance", value=1.0)
    s = SpectrumFactory.converter("radiance")(ureg.Quantity(1e6, "W/km^2/sr/nm"))
    assert s == UniformSpectrum(quantity="radiance", value=1.0)
    with pytest.raises(pinttr.exceptions.UnitsError):
        SpectrumFactory.converter("irradiance")(ureg.Quantity(1, "W/m^2/sr/nm"))


def test_uniform(mode_mono):
    from eradiate.kernel.core.xml import load_dict

    # Instantiate without argument
    UniformSpectrum()

    # Instantiate with only quantity
    UniformSpectrum(quantity=PhysicalQuantity.COLLISION_COEFFICIENT)
    UniformSpectrum(quantity="collision_coefficient")

    # Instantiate with unsupported quantity
    with pytest.raises(ValueError):
        UniformSpectrum(quantity="speed")

    # Instantiate with all arguments
    s = UniformSpectrum(quantity="collision_coefficient", value=1.)
    assert s.value == ureg.Quantity(1., "m^-1")
    with pytest.raises(ValueError):
        UniformSpectrum(quantity="collision_coefficient", value=-1.)

    # Raise if units and quantity are inconsistent
    with pytest.raises(pinttr.exceptions.UnitsError):
        UniformSpectrum(
            quantity="collision_coefficient",
            value=ureg.Quantity(1., "")
        )

    # Instantiate from dictionary
    s = UniformSpectrum.from_dict({
        "quantity": "radiance", "value": 1., "value_units": "W/km^2/sr/nm"
    })

    # Produced kernel dict is valid
    assert load_dict(onedict_value(s.kernel_dict())) is not None

    # Unit scaling is properly applied
    with ucc.override({"radiance": "W/m^2/sr/nm"}):
        s = UniformSpectrum(quantity="radiance", value=1.)
    with uck.override({"radiance": "kW/m^2/sr/nm"}):
        d = s.kernel_dict()
        assert np.allclose(d["spectrum"]["value"], 1e-3)


def test_solar(mode_mono):
    from eradiate.kernel.core.xml import load_dict

    # We can instantiate the element
    s = SolarIrradianceSpectrum()

    # Unsupported solar spectrum keywords raise
    with pytest.raises(ValueError):
        SolarIrradianceSpectrum(dataset="doesnt_exist")

    # Produced kernel dict is valid
    assert load_dict(onedict_value(s.kernel_dict())) is not None

    # A more detailed specification still produces a valid object
    s = SolarIrradianceSpectrum(scale=2.0)
    assert load_dict(onedict_value(s.kernel_dict())) is not None

    # Element doesn't work out of the supported spectral range
    s = SolarIrradianceSpectrum(dataset="thuillier_2003")

    with pytest.raises(ValueError):
        eradiate.set_mode("mono", wavelength=2400.)
        s.kernel_dict()

    # solid_2017_mean dataset can be used
    eradiate.set_mode("mono", wavelength=550.)
    s = SolarIrradianceSpectrum(dataset="solid_2017_mean")
    assert load_dict(onedict_value(s.kernel_dict()))

    # values properties interpolates the irradiance as expected
    eradiate.set_mode("mono", wavelength=550.)
    s = SolarIrradianceSpectrum(dataset="thuillier_2003")
    assert np.allclose(s.values, ureg.Quantity(1.87938, "W/m^2/nm"))
