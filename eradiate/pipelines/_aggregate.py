import typing as t

import attr
import xarray as xr

from ._core import PipelineStep
from ..attrs import parse_docs
from ..scenes.measure import Measure


@parse_docs
@attr.s
class AggregateSampleCount(PipelineStep):
    """
    Aggregate sample count.

    This post-processing pipeline step aggregates sample counts:

    * it computes the average of sensor values weighted by the sample count;
    * it sums the ``spp`` dimension.

    The ``spp_index`` dimension is dropped during this step and the ``spp``
    variable ends up with no dimension.
    """

    def transform(self, x: t.Any) -> t.Any:
        with xr.set_options(keep_attrs=True):
            result = x.weighted(x.spp).mean(dim="spp_index")
            result["spp"] = x.spp.sum()

        return result


@attr.s
class AggregateCKDQuad(PipelineStep):
    """
    Compute CKD quadrature.
    """

    measure: Measure = attr.ib()

    def transform(self, x: t.Any) -> t.Any:
        # TODO
        # If not in CKD mode, no-op
        # Otherwise, compute quadrature spectrum-indexed variables and turn spp
        # into a per-bin average
        # Final output has a spectral coordinate but retains bin ID coordinate
        raise NotImplementedError
