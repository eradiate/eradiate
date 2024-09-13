from __future__ import annotations

import enum
import warnings

import attrs
import pint
import xarray as xr

from ..attrs import documented, frozen
from ..quad import Quad, QuadType
from ..radprops import CKDAbsorptionDatabase


class CKDQuadPolicy(enum.Enum):
    """
    This enumeration defines flags mapping to policies defining the number of
    CKD spectral quadrature points for a given spectral bin.

    The ``"fixed"`` policy uses a constant number of quadrature points,
    regardless of the spectral bin. The other policies will determine a number
    of quadrature point based on molecular absorption data and an error
    criterion, seeking to minimize or achieve an error on the total column
    transmittance w.r.t. absorption for each spectral bin.
    """

    FIXED = "fixed"  #: Fixed number of quadrature points.
    MINIMIZE_ERROR = "minimize_error"
    """Number of quadrature points that minimizes the error in each spectral bin
    (not implemented yet)."""
    ERROR_THRESHOLD = "error_threshold"
    """Optimal number of quadrature points that achieves the error threshold
    (not implemented yet)."""


@frozen
class CKDQuadConfig:
    """
    This class holds configuration parameters for a CKD quadrature rule. Once
    the quadrature definition is set, it can query an absorption database to
    generate a quadrature rule for a specified spectral bin using its
    :meth:`.get_quad` method.
    """

    type: QuadType = documented(
        attrs.field(default="gauss_legendre", converter=QuadType),
        doc="Quadrature type.",
        type=".QuadType",
        init_type=".QuadType or str",
        default='"gauss_legendre"',
    )

    ng_max: int = documented(
        attrs.field(default=1, converter=int),
        doc="Maximum number of quadrature points.",
        type="int",
        default="1",
    )

    policy: CKDQuadPolicy = documented(
        attrs.field(default="fixed", converter=CKDQuadPolicy),
        doc="Quadrature definition policy.",
        type=".CKDQuadPolicy",
        init_type=".CKDQuadPolicy or str",
        default="fixed",
    )

    @classmethod
    def convert(self, value) -> CKDQuadConfig:
        """
        Convert a value to a :class:`.CKDQuadConfig`. If ``value`` is a
        dictionary, its values are passed to the constructor as keyword
        arguments. Otherwise, ``value`` is returned unchanged.
        """
        if isinstance(value, dict):
            return CKDQuadConfig(**value)
        else:
            return value

    def get_quad(
        self,
        abs_db: CKDAbsorptionDatabase | None = None,
        wcenter: pint.Quantity | float | None = None,
    ):
        """
        Generate a quadrature rule for a specific bin, using information from a
        specified molecular absorption database.

        Parameters
        ----------
        abs_db : .CKDAbsorptionDatabase, optional
            An absorption database with a transmittance error variable.

        wcenter : quantity or float, optional
            The central wavelength of the spectral bin for which the quadrature
            rule is requested. Unitless values are interpreted in default
            wavelength units (generally nm).

        Returns
        -------
        .CKDQuad
        """
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
