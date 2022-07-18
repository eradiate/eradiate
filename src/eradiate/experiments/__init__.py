from ..util import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submod_attrs={
        "_core": ["EarthObservationExperiment", "Experiment", "mitsuba_run", "run"],
        "_atmosphere": ["AtmosphereExperiment", "OneDimExperiment"],
        "_canopy": ["CanopyExperiment", "RamiExperiment"],
        "_rami4atm": ["Rami4ATMExperiment"],
    },
)

del lazy_loader
