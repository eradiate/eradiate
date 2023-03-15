# Basic pipeline infrastructure
# Aggregate steps
from ._aggregate import AggregateCKDQuad as AggregateCKDQuad
from ._aggregate import AggregateRadiosity as AggregateRadiosity

# Assemble steps
from ._assemble import AddIllumination as AddIllumination
from ._assemble import AddSpectralResponseFunction as AddSpectralResponseFunction
from ._assemble import AddViewingAngles as AddViewingAngles

# Compute steps
from ._compute import ApplySpectralResponseFunction as ApplySpectralResponseFunction
from ._compute import ComputeAlbedo as ComputeAlbedo
from ._compute import ComputeReflectance as ComputeReflectance

from ._core import ApplyCallable as ApplyCallable  # isort: skip
from ._core import Pipeline as Pipeline  # isort: skip
from ._core import PipelineStep as PipelineStep  # isort: skip

# Gather step
from ._gather import Gather as Gather
