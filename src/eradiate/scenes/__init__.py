from ..util import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules=[
        "atmosphere",
        "biosphere",
        "bsdfs",
        "core",
        "illumination",
        "integrators",
        "measure",
        "phase",
        "shapes",
        "spectra",
        "surface",
    ],
)

del lazy_loader
