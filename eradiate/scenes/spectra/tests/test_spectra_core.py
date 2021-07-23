import pinttr
import pytest

from eradiate import unit_registry as ureg
from eradiate.scenes.spectra import UniformSpectrum, spectrum_factory


def test_converter(modes_all_mono):
    # Dicts are correctly processed
    s = spectrum_factory.converter("radiance")({"type": "uniform"})
    assert s == UniformSpectrum(quantity="radiance", value=1.0)
    s = spectrum_factory.converter("irradiance")({"type": "uniform"})
    assert s == UniformSpectrum(quantity="irradiance", value=1.0)

    # Floats and quantities are correctly processed
    s = spectrum_factory.converter("radiance")(1.0)
    assert s == UniformSpectrum(quantity="radiance", value=1.0)
    s = spectrum_factory.converter("radiance")(ureg.Quantity(1e6, "W/km^2/sr/nm"))
    assert s == UniformSpectrum(quantity="radiance", value=1.0)
    with pytest.raises(pinttr.exceptions.UnitsError):
        spectrum_factory.converter("irradiance")(ureg.Quantity(1, "W/m^2/sr/nm"))
