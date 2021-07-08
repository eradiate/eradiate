from ._air_scattering_coefficient import AirScatteringCoefficientSpectrum
from ._core import Spectrum, spectrum_factory
from ._interpolated import InterpolatedSpectrum
from ._solar_irradiance import SolarIrradianceSpectrum
from ._uniform import UniformSpectrum

__all__ = [
    "Spectrum",
    "spectrum_factory",
    "AirScatteringCoefficientSpectrum",
    "SolarIrradianceSpectrum",
    "UniformSpectrum",
    "InterpolatedSpectrum",
]
