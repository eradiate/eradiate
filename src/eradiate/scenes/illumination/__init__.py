from ...util import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submod_attrs={
        "_core": ["Illumination", "illumination_factory"],
        "_constant": ["ConstantIllumination"],
        "_directional": ["DirectionalIllumination"],
    },
)

del lazy_loader
