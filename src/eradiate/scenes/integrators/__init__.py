from ...util import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submod_attrs={
        "_core": ["Integrator", "integrator_factory"],
        "_path_tracers": [
            "PathIntegrator",
            "VolPathIntegrator",
            "VolPathMISIntegrator",
        ],
    },
)

del lazy_loader
