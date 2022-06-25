from ...util import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submod_attrs={
        "_core": ["Spectrum", "spectrum_factory"],
        "_air_scattering_coefficient": ["AirScatteringCoefficientSpectrum"],
        "_interpolated": ["InterpolatedSpectrum"],
        "_solar_irradiance": ["SolarIrradianceSpectrum"],
        "_uniform": ["UniformSpectrum"],
    },
)

del lazy_loader
