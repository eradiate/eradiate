from ..util import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submod_attrs={
        # Basic pipeline infrastructure
        "_core": ["Pipeline", "PipelineStep"],
        # Gather step
        "_gather": ["Gather"],
        # Aggregate steps
        "_aggregate": [
            "AggregateCKDQuad",
            "AggregateRadiosity",
            "AggregateSampleCount",
        ],
        # Assemble steps
        "_assemble": [
            "AddIllumination",
            "AddSpectralResponseFunction",
            "AddViewingAngles",
        ],
        # Compute steps
        "_compute": [
            "ApplySpectralResponseFunction",
            "ComputeAlbedo",
            "ComputeReflectance",
        ],
    },
)

del lazy_loader
