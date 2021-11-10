from ._aggregate import AggregateCKDQuad, AggregateRadiosity, AggregateSampleCount
from ._assemble import AddIllumination, AddSpectralResponseFunction, AddViewingAngles
from ._compute import ApplySpectralResponseFunction, ComputeAlbedo, ComputeReflectance
from ._core import Pipeline, PipelineStep
from ._gather import Gather

__all__ = [
    # Basic pipeline infrastructure
    "Pipeline",
    "PipelineStep",
    # Gather step
    "Gather",
    # Aggregate steps
    "AggregateCKDQuad",
    "AggregateRadiosity",
    "AggregateSampleCount",
    # Assemble steps
    "AddIllumination",
    "AddSpectralResponseFunction",
    "AddViewingAngles",
    # Compute steps
    "ApplySpectralResponseFunction",
    "ComputeReflectance",
    "ComputeAlbedo",
]
