from __future__ import annotations

import math

import attrs
import mitsuba as mi
import numpy as np
import pint
import pinttrs

from ._core import Measure
from ... import validators
from ...attrs import define, documented
from ...units import symbol
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg

#: Normalised image radius against the angle from the optical axis, published
#: in the result metadata.
_PROJECTION_FORMULAS = {
    "equidistant": "r(theta) = theta / theta_max, theta_max = fov/2",
    "equisolid": "r(theta) = sin(theta/2) / sin(theta_max/2), theta_max = fov/2",
    "stereographic": ("r(theta) = tan(theta/2) / tan(theta_max/2), theta_max = fov/2"),
    "orthographic": "r(theta) = sin(theta) / sin(theta_max), theta_max = fov/2",
    "polynomial": (
        "r(theta) = c0*theta + c1*theta**2 + ... with theta in radian, the "
        "coefficients given by 'lens_coefficients', and the fit valid over "
        "[0, fov/2]"
    ),
    "equisolid_full": (
        "square-to-hemisphere map; ignores fov, centre and radius, and images "
        "every film sample"
    ),
}


def _convert_lens_coefficients(value) -> list[float]:
    """
    Coerce polynomial coefficients to a list of ``float``, accepting a scalar for
    a single-coefficient calibration. Strings are refused: iterating one yields
    characters, and the resulting complaint names neither the field nor the
    problem.
    """
    if isinstance(value, str):
        raise TypeError(
            f"lens_coefficients must be a number or a sequence of numbers, "
            f"got the string {value!r}"
        )

    result = np.atleast_1d(np.asarray(value, dtype=float))

    if result.ndim != 1:
        raise ValueError(
            f"lens_coefficients must be one-dimensional, got shape {result.shape}"
        )

    return result.tolist()


@define(eq=False, slots=False)
class FisheyeCameraMeasure(Measure):
    """
    Fisheye camera scene element [``fisheye``].

    This scene element is a thin wrapper around the ``fisheye`` sensor kernel
    plugin. It positions a circular fisheye camera based on a set of vectors
    specifying the origin, viewing direction and 'up' direction of the camera.

    Results carry a ``valid`` mask flagging the pixels that image a direction,
    and the projection is published on the dataset attributes. Per-pixel viewing
    angles are not computed.
    """

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    spp: int = documented(
        attrs.field(default=32, converter=int, validator=validators.is_positive),
        doc="Number of samples per pixel.",
        type="int",
        default="32",
    )

    _film_resolution: tuple[int, int] = documented(
        attrs.field(
            default=(32, 32),
            converter=tuple,
            validator=attrs.validators.deep_iterable(
                member_validator=[
                    attrs.validators.instance_of(int),
                    attrs.validators.gt(0),
                ],
                iterable_validator=validators.has_len(2),
            ),
        ),
        doc="Film resolution as a (width, height) 2-tuple.",
        type="tuple of int",
        init_type="array-like",
        default="(32, 32)",
    )

    @property
    def film_resolution(self) -> tuple[int, int]:
        return self._film_resolution

    origin: pint.Quantity = documented(
        pinttrs.field(
            factory=lambda: [1, 1, 1] * ureg.m,
            validator=[validators.has_len(3), pinttrs.validators.has_compatible_units],
            units=ucc.deferred("length"),
        ),
        doc="A 3-vector specifying the position of the camera.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="quantity",
        init_type="array-like",
        default="[1, 1, 1] m",
    )

    target: pint.Quantity = documented(
        pinttrs.field(
            factory=lambda: [0, 0, 0] * ureg.m,
            validator=[validators.has_len(3), pinttrs.validators.has_compatible_units],
            units=ucc.deferred("length"),
        ),
        doc="Point location targeted by the camera.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="quantity",
        init_type="array-like",
        default="[0, 0, 0] m",
    )

    @target.validator
    @origin.validator
    def _target_origin_validator(self, attribute, value):
        if np.allclose(self.target, self.origin):
            raise ValueError(
                f"While initializing '{attribute.name}': "
                f"Origin and target must not be equal, "
                f"got target = {self.target}, origin = {self.origin}"
            )

    up: np.ndarray = documented(
        attrs.field(
            factory=lambda: [0, 0, 1],
            converter=np.array,
            validator=validators.has_len(3),
        ),
        doc="A 3-vector specifying the up direction of the camera; it points to "
        "the top of the image.\n"
        "This vector must not be colinear with the camera's viewing direction,\n"
        "which is given by ``target - origin``.",
        type="array",
        default="[0, 0, 1]",
    )

    @up.validator
    def _up_validator(self, attribute, value):
        direction = self.target - self.origin
        if np.allclose(np.cross(direction, value), 0):
            raise ValueError(
                f"While initializing '{attribute.name}': "
                f"up direction must not be colinear with viewing direction, "
                f"got up = {self.up}, direction = {direction}"
            )

    projection_model: str = documented(
        attrs.field(
            default="equisolid",
            converter=str,
            validator=attrs.validators.in_(
                (
                    "equidistant",
                    "equisolid",
                    "stereographic",
                    "orthographic",
                    "polynomial",
                    # Developer-only mode, undocumented here and in the plugin
                    "equisolid_full",
                )
            ),
        ),
        doc="Projection model, must be one of: equidistant | equisolid | "
        "stereographic | orthographic | polynomial.\n"
        "\n"
        "The first four are analytic lens models, for which ``fov`` maps to the "
        "image circle. The ``polynomial`` model instead uses a calibrated "
        "*normalised* projection function ``rho(theta) = c0 * theta + c1 * "
        "theta**2 + ...``, of however many coefficients are supplied; in that "
        "mode ``fov`` declares the angular domain the calibration was fitted "
        "over. See ``lens_coefficients`` and ``fov``.",
        type="str",
        default="equisolid",
    )

    fov: pint.Quantity = documented(
        pinttrs.field(default=180.0 * ureg.deg, units=ureg.deg),
        doc="Camera field of view, i.e. the full angle subtended by the image "
        "circle. For ``projection_model == 'polynomial'`` it instead declares "
        "the angular domain ``[0, fov/2]`` the calibration was fitted over, "
        "e.g. ``fov=220`` for a calibration fitted to 110°.\n"
        "\n"
        "Unit-enabled field (default: degree).",
        type="quantity",
        init_type="quantity or float",
        default="180°",
    )

    lens_coefficients: list[float] | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(_convert_lens_coefficients),
        ),
        doc="Coefficients of the polynomial projection formula (see "
        "``projection_model``) in ascending order, i.e. ``[c0, c1, ...]`` for "
        "``rho(theta) = c0 * theta + c1 * theta**2 + ...``. The polynomial "
        "order is the number of coefficients supplied. Required when "
        "``projection_model == 'polynomial'``; unused otherwise.\n"
        "\n"
        "The edge value ``rho(fov/2)`` decides what the rim shows: below 1 the "
        "projection stops short of the image circle and the ring beyond it "
        "images nothing; above 1 the circle fills before ``fov`` is reached, so "
        "the image spans less than ``fov``. It is published as ``rho_max`` in "
        "the result attributes.\n"
        "\n"
        "The kernel inverts the calibration when the scene is built and refuses "
        "one it cannot invert reliably.",
        type="list of float or None",
        init_type="float or array-like, optional",
        default="None",
    )

    image_circle_radius: float | None = documented(
        attrs.field(default=None, converter=attrs.converters.optional(float)),
        doc="Radius of the image circle in pixels at "
        "``calibration_resolution``. This is the pixel scale of the projection: "
        "the radius that a normalised image radius of ``rho = 1`` corresponds "
        "to, not necessarily the radius at which the image ends, see "
        "``lens_coefficients``. The image circle describes the lens mounting "
        "rather than the projection, so this applies to every "
        "``projection_model``. Kernel defaults to the disk inscribed in the "
        "film if unset.",
        type="float or None",
        init_type="float, optional",
        default="None",
    )

    center_x: float | None = documented(
        attrs.field(default=None, converter=attrs.converters.optional(float)),
        doc="Optical-centre x coordinate in pixels at "
        "``calibration_resolution``. Applies to every ``projection_model``. "
        "Kernel defaults to the film centre if unset.",
        type="float or None",
        init_type="float, optional",
        default="None",
    )

    center_y: float | None = documented(
        attrs.field(default=None, converter=attrs.converters.optional(float)),
        doc="Optical-centre y coordinate in pixels at "
        "``calibration_resolution``. Applies to every ``projection_model``. "
        "Kernel defaults to the film centre if unset.",
        type="float or None",
        init_type="float, optional",
        default="None",
    )

    calibration_resolution: tuple[int, int] | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(tuple),
            validator=attrs.validators.optional(
                attrs.validators.deep_iterable(
                    member_validator=[
                        attrs.validators.instance_of(int),
                        attrs.validators.gt(0),
                    ],
                    iterable_validator=validators.has_len(2),
                )
            ),
        ),
        doc="Resolution (width, height) in which calibration data are expressed "
        "(pixel frame for ``center_x``, ``center_y``, ``image_circle_radius``). "
        "Defaults to ``film_resolution`` if unset; aspect ratio must match. "
        "Applies to every ``projection_model``.",
        type="tuple of int or None",
        init_type="array-like, optional",
        default="None",
    )

    far_clip: pint.Quantity = documented(
        pinttrs.field(
            default=1e4 * ureg.km,
            units=ucc.deferred("length"),
        ),
        doc="Distance to the far clip sphere: rays are traced up to this radial "
        "distance from the camera.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="quantity",
        init_type="quantity of float",
        default="10 000 km",
    )

    @property
    def _image_circle(self) -> tuple[float, float, float]:
        """
        Normalised image circle as ``(center_x, center_y, radius)``: the centre
        in film coordinates and the radius as a fraction of the film width --
        the kernel scales its axes by ``(1/radius, aspect/radius)``, so the
        radius is expressed in units of width whatever the film shape. Unset
        fields fall back to the kernel plugin's own defaults: the film centre,
        and the disk inscribed in the film.
        """
        width, height = self.film_resolution
        calib_w, calib_h = self.calibration_resolution or self.film_resolution

        center_x = 0.5 if self.center_x is None else self.center_x / calib_w
        center_y = 0.5 if self.center_y is None else self.center_y / calib_h
        radius = (
            0.5 * min(width, height) / width
            if self.image_circle_radius is None
            else self.image_circle_radius / calib_w
        )

        return center_x, center_y, radius

    @property
    def _rho_max(self) -> float:
        """
        Normalised image radius the projection reaches at ``fov/2``. Below 1 for
        a calibration that stops short of the image circle, in which case the
        ring beyond it images nothing.
        """
        if self.projection_model != "polynomial":
            return 1.0

        theta_max = 0.5 * self.fov.m_as(ureg.rad)
        return float(np.polyval(self.lens_coefficients[::-1] + [0.0], theta_max))

    @property
    def valid_mask(self) -> np.ndarray:
        """
        ndarray: Boolean ``(height, width)`` array, ``True`` where the pixel
        images a direction. Stored in bitmap layout so that it aligns with the
        recorded variable.

        Notes
        -----
        This reproduces the kernel plugin's own test, ``r <= min(1, rho_max)``,
        which needs the image circle but not the projection inverse -- so it
        stays closed form for every model, and is pinned against the kernel by
        the unit tests.
        """
        width, height = self.film_resolution

        if self.projection_model == "equisolid_full":
            # A square-to-hemisphere map: every film sample images a direction
            return np.ones((height, width), dtype=bool)

        center_x, center_y, radius = self._image_circle
        u = (np.arange(width) + 0.5) / width
        v = (np.arange(height) + 0.5) / height
        q_x = (center_x - u)[np.newaxis, :] / radius
        q_y = (center_y - v)[:, np.newaxis] * (height / width) / radius

        return np.hypot(q_x, q_y) <= min(1.0, self._rho_max)

    @property
    def projection_metadata(self) -> dict:
        """
        dict: Description of the projection, published on the result's
        dataset-level attributes. The measure reports no per-pixel viewing
        angles; these are the values needed to compute them. Strings, numbers
        and lists of numbers only, so that the result stays serialisable.

        The keys are ``projection_model``, ``projection_formula``, ``fov``,
        ``fov_units``, ``film_width``, ``film_height``, ``center_x``,
        ``center_y``, ``image_circle_radius`` and ``rho_max``, plus
        ``lens_coefficients`` for the polynomial model.
        """
        width, height = self.film_resolution
        center_x, center_y, radius = self._image_circle

        result = {
            "projection_model": self.projection_model,
            "projection_formula": _PROJECTION_FORMULAS[self.projection_model],
            "fov": float(self.fov.m_as(ureg.deg)),
            "fov_units": "deg",
            "film_width": int(width),
            "film_height": int(height),
            "center_x": float(center_x),
            "center_y": float(center_y),
            "image_circle_radius": float(radius),
            "rho_max": float(self._rho_max),
        }

        if self.projection_model == "polynomial":
            result["lens_coefficients"] = [float(x) for x in self.lens_coefficients]

        return result

    def update(self) -> None:
        # Inherit docstring
        super().update()

        if self.projection_model == "polynomial":
            if not self.lens_coefficients:
                raise ValueError(
                    "the 'polynomial' projection model requires at least one "
                    "coefficient in 'lens_coefficients'"
                )
        elif self.lens_coefficients is not None:
            raise ValueError(
                f"'lens_coefficients' is only used by the 'polynomial' "
                f"projection model, got "
                f"projection_model = '{self.projection_model}'"
            )

        # center_x, center_y and image_circle_radius are pixel counts in the
        # calibration frame, so that frame must match the film's shape for the
        # normalisation to mean anything
        if self.calibration_resolution is not None:
            calib_w, calib_h = self.calibration_resolution
            film_w, film_h = self.film_resolution
            # Calibration sheets are hand-measured, so allow small measurement error
            if not math.isclose(calib_w / calib_h, film_w / film_h, rel_tol=1e-3):
                raise ValueError(
                    f"film_resolution {self.film_resolution} and "
                    f"calibration_resolution {(calib_w, calib_h)} must have the "
                    f"same aspect ratio"
                )

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def kernel_type(self) -> str:
        # Inherit docstring
        return "fisheye"

    @property
    def template(self) -> dict:
        # Inherit docstring
        result = super().template

        result["far_clip"] = self.far_clip.m_as(uck.get("length"))
        result["projection_model"] = self.projection_model

        center_x, center_y, radius = self._image_circle
        if self.center_x is not None:
            result["center_x"] = center_x
        if self.center_y is not None:
            result["center_y"] = center_y
        if self.image_circle_radius is not None:
            result["radius"] = radius

        if self.projection_model == "polynomial":
            # Sent as text
            result["lens_coefficients"] = ", ".join(map(repr, self.lens_coefficients))

        # Image-circle rim for the analytic models, calibrated domain for the
        # polynomial one
        result["fov"] = self.fov.m_as(ureg.deg)

        target = self.target.m_as(uck.get("length"))
        origin = self.origin.m_as(uck.get("length"))
        result["to_world"] = mi.ScalarTransform4f().look_at(
            origin=origin, target=target, up=self.up
        )

        return result

    # --------------------------------------------------------------------------
    #                        Post-processing information
    # --------------------------------------------------------------------------

    @property
    def var(self) -> tuple[str, dict]:
        return "radiance", {
            "standard_name": "radiance",
            "long_name": "radiance",
            "units": symbol(uck.get("radiance")),
        }
