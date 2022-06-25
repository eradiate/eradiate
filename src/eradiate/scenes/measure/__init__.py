from ...util import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submod_attrs={
        "_core": ["Measure", "MeasureSpectralConfig", "measure_factory"],
        "_target": ["Target", "TargetPoint", "TargetRectangle"],
        "_distant_flux": ["DistantFluxMeasure"],
        "_hemispherical_distant": ["HemisphericalDistantMeasure"],
        "_multi_distant": ["MultiDistantMeasure"],
        "_multi_radiancemeter": ["MultiRadiancemeterMeasure"],
        "_perspective": ["PerspectiveCameraMeasure"],
        "_radiancemeter": ["RadiancemeterMeasure"],
    },
)

del lazy_loader
