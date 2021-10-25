import typing as t
import warnings

import attr
import numpy as np
import pint
import pinttr
import xarray as xr

from ._core import PipelineStep
from .. import unit_registry as ureg
from ..attrs import documented, parse_docs
from ..frame import angles_in_hplane
from ..scenes.illumination import DirectionalIllumination
from ..scenes.measure import Measure
from ..units import symbol
from ..units import unit_context_config as ucc


@parse_docs
@attr.s
class AddIllumination(PipelineStep):
    """
    Add illumination data.

    This post-processing pipeline step adds Sun angle coordinates and
    illumination data:

    * the ``sza`` (Sun zenith angle) and ``saa`` (Sun azimuth angle) scalar
      coordinates are defined and attached to all existing variables;
    * an ``illumination`` data variable is created, with dimensions ``sza`` and
      ``vaa``.
    """

    illumination: DirectionalIllumination = documented(
        attr.ib(
            validator=attr.validators.instance_of(DirectionalIllumination),
            repr=lambda self: f"{self.__class__.__name__}(id='{self.id}', ...)",
        ),
        doc="A :class:`.DirectionalIllumination` instance from which the "
        "illumination data originates.",
        type=":class:`.DirectionalIllumination`",
    )

    def transform(self, x: t.Any) -> t.Any:
        # TODO
        raise NotImplementedError


@attr.s
class AddViewingAngles(PipelineStep):
    """
    Create new ``vza`` and ``vaa`` coordinate variables mapping viewing angles
    to other coordinates.
    """

    measure: Measure = documented(
        attr.ib(
            validator=attr.validators.instance_of(Measure),
            repr=lambda self: f"{self.__class__.__name__}(id='{self.id}', ...)",
        ),
        doc="A :class:`.Measure` instance from which the processed data originates.",
        type=":class:`.Measure`",
    )

    def transform(self, x: t.Any) -> t.Any:
        measure = self.measure
        viewing_angles = measure.viewing_angles

        # Collect zenith and azimuth values
        theta = viewing_angles[:, 0]
        phi = viewing_angles[:, 1]

        if measure.hplane is not None:
            theta, phi = _remap_viewing_angles_plane(measure.hplane, theta, phi)

        with xr.set_options(keep_attrs=True):
            result = x.assign_coords(
                {
                    "vza": (
                        ("x_index", "y_index"),
                        theta.m_as(ureg.deg).reshape((-1, 1)),
                        {
                            "standard_name": "viewing_zenith_angle",
                            "long_name": "viewing zenith angle",
                            "units": symbol("deg"),
                        },
                    ),
                    "vaa": (
                        ("x_index", "y_index"),
                        phi.m_as(ureg.deg).reshape((-1, 1)),
                        {
                            "standard_name": "viewing_azimuth_angle",
                            "long_name": "viewing azimuth angle",
                            "units": symbol("deg"),
                        },
                    ),
                }
            )

        return result


@ureg.wraps(ret=("deg", "deg"), args=("deg", "deg", "deg"), strict=True)
def _remap_viewing_angles_plane(
    plane: np.typing.ArrayLike,
    theta: np.typing.ArrayLike,
    phi: np.typing.ArrayLike,
) -> t.Tuple[np.typing.ArrayLike, np.typing.ArrayLike]:
    r"""
    Remap viewing angles to a hemisphere plane cut.

    Parameters
    ----------
    plane : quantity
         Plane cut orientation (scalar value).

    theta : quantity
        List of zenith angle values with (N,) shape.

    phi : quantity
        List of azimuth angle values with (N,) shape.

    Returns
    -------
    theta : quantity
        List of zenith angle values in :math:`[-\pi, \pi]` with (N,) shape.

    phi : quantity
        List of azimuth angle values in :math:`[0, 2\pi]` with (N,) shape
        (equal to `plane` modulo :math:`\pi`).

    Warns
    -----

    """
    # Normalise all angles
    plane = plane % 360.0
    theta = theta % 360.0
    phi = phi % 360.0

    # Check that phi values are compatible with requested plane
    in_plane_positive, in_plane_negative = angles_in_hplane(plane, theta, phi)

    # Check if any point is allocated to both half-planes (uncomment to debug)
    # assert not np.any(in_plane_positive & in_plane_negative)

    # Normalise zenith values
    theta = np.where(in_plane_positive, theta, -theta)

    # Normalise azimuth values
    phi = np.full_like(theta, plane)

    # Check ordering and warn if it is not strictly increasing
    if not _is_sorted(theta):
        warnings.warn(
            "Viewing zenith angle values are sorted sorted in ascending order, "
            "you might want to consider changing direction definitions."
        )

    return theta, phi


_is_sorted = lambda a: np.all(a[:-1] <= a[1:])
