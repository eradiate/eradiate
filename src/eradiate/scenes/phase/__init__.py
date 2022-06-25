from ...util import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submod_attrs={
        "_blend": ["BlendPhaseFunction"],
        "_core": ["PhaseFunction", "phase_function_factory"],
        "_hg": ["HenyeyGreensteinPhaseFunction"],
        "_isotropic": ["IsotropicPhaseFunction"],
        "_rayleigh": ["RayleighPhaseFunction"],
        "_tabulated": ["TabulatedPhaseFunction"],
    },
)

del lazy_loader
