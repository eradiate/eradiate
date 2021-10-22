import attr

from ._core import PipelineStep
from ..attrs import parse_docs


@parse_docs
@attr.s
class ComputeReflectance(PipelineStep):
    """
    Compute reflectance data.
    """
