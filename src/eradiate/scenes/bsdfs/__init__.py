from ...util import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submod_attrs={
        "_core": ["BSDF", "bsdf_factory"],
        "_black": ["BlackBSDF"],
        "_checkerboard": ["CheckerboardBSDF"],
        "_lambertian": ["LambertianBSDF"],
        "_rpv": ["RPVBSDF"],
    },
)

del lazy_loader
