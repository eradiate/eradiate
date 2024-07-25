from __future__ import annotations

import enum
import warnings

import attrs
import xarray as xr

from ..attrs import define
from ..quad import Quad, QuadType


class CKDQuadPolicy(enum.Enum):
    FIXED = "fixed"
    MINIMIZE_ERROR = "minimize_error"
    ERROR_THRESHOLD = "error_threshold"


@define
class CKDQuadConfig:
    type: QuadType = attrs.field(default="gauss_legendre", converter=QuadType)
    ng_max: int = attrs.field(default=16, converter=int)
    policy: CKDQuadPolicy = attrs.field(default="fixed", converter=CKDQuadPolicy)

    def get_quad(self, abs_db=None, wcenter=None):
        if abs_db is None:
            # If no spectral information is passed, use default policy
            ng = self.ng_max
        else:
            if self.policy is not CKDQuadPolicy.FIXED:
                # TODO: Temporary, implement advanced quadrature point count
                #  policies when sample count allocation is refactored
                warnings.warn(
                    "An advanced spectral quadrature configuration was defined, "
                    "but this feature is currently disabled. Falling back to a "
                    f"fixed quadrature point policy with {self.ng_max} g-points."
                )
            ng = self.ng_max

        return Quad.new(type=self.type, n=ng)


# TODO: Refactor to use new absorption database format
def ng_minimum(error: xr.DataArray, ng_max: int | None = None):
    """
    Find the number of quadrature points that minimizes the error.

    Parameters
    ----------
    error : DataArray
        Error data.

    ng_max : int, optional
        Maximum number of quadrature points. If not provided, it will be
        inferred from the error data.

    Returns
    -------
    int
        Number of quadrature points that minimizes the error.
    """

    if ng_max is None:
        ng_max = int(error.ng.max())

    error_w0 = error.isel(w=0)
    ng_min = int(error.ng.where(error_w0 == error_w0.min(), drop=True)[0])
    return ng_max if ng_min > ng_max else ng_min


# TODO: Refactor to use new absorption database format
def ng_threshold(error: xr.DataArray, threshold: float, ng_max: int | None = None):
    """
    Find the number of quadrature points so that the error is (strictly) below
    a specified threshold value.

    Parameters
    ----------
    error : DataArray
        Error data.

    threshold : float
        Error threshold.

    ng_max : int, optional
        Maximum number of quadrature points. If not provided, it will be
        inferred from the error data.

    Returns
    -------
    int
        Number of quadrature points so that the error is below the threshold.
    """

    if ng_max is None:
        ng_max = int(error.ng.max())

    error_w0 = error.isel(w=0)
    ng = error.ng.where(error_w0 < threshold, drop=True)

    if ng.size == 0:
        return ng_max
    else:
        ng = int(ng[0])
        return ng_max if ng > ng_max else ng
