from ._core import Spectrum, SpectrumFactory
from ._solar_irradiance import SolarIrradianceSpectrum
from ._uniform import UniformSpectrum
from ._air_scattering_coefficient import AirScatteringCoefficientSpectrum

__all__ = [
    "Spectrum",
    "SpectrumFactory",
    "AirScatteringCoefficientSpectrum",
    "SolarIrradianceSpectrum",
    "UniformSpectrum",
]
