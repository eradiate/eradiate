from ...util import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submod_attrs={
        "_core": ["Surface", "surface_factory"],
        "_basic": ["BasicSurface"],
        "_central_patch": ["CentralPatchSurface"],
        "_dem": ["DEMSurface"],
    },
)

del lazy_loader
