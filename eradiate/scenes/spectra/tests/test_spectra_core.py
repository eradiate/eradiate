import pinttr
import pytest

from eradiate import unit_registry as ureg
from eradiate.scenes.spectra import SpectrumFactory, UniformSpectrum


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
