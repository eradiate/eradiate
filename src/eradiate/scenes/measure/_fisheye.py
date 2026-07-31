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


@define(eq=False, slots=False)
class FisheyeCameraMeasure(Measure):
    """
    Fisheye camera scene element [``fisheye``].

    This scene element is a thin wrapper around the ``fisheye`` sensor kernel
    plugin. It positions a circular fisheye camera based on a set of vectors
    specifying the origin, viewing direction and 'up' direction of the camera.
    The film centre images the viewing direction and the image circle maps the
    field of view; film samples outside the image circle produce black pixels.

    The camera's ``up`` vector points to the top of the image. The canonical
    upward-looking configuration for hemispherical photography is::

        FisheyeCameraMeasure(
            origin=[0, 0, 0.5] * ureg.m,
            target=[0, 0, 1] * ureg.m,
            up=[0, 1, 0],
        )

    The ``polynomial`` projection model reproduces a real, calibrated
    camera/lens combination pixel for pixel. Calibration data are passed in the
    native units of the calibration sheet (pixels at ``calibration_resolution``)
    and are converted to the normalised quantities expected by the kernel::

        FisheyeCameraMeasure(
            projection_model="polynomial",
            lens_a=0.7195,
            lens_b=-0.0637,
            center_x=3017,          # 'Lens X'
            center_y=1989,          # 'Lens Y'
            max_radius=1643,        # 'Maximum Radius'
            calibration_resolution=(6000, 4000),
            film_resolution=(600, 400),  # any same-aspect resolution
        )
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
                member_validator=attrs.validators.instance_of(int),
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
                f"While initializing {attribute}: "
                f"Origin and target must not be equal, "
                f"got target = {self.target}, origin = {self.origin}"
            )

    up: np.ndarray = documented(
        attrs.field(
            factory=lambda: [0, 0, 1],
            converter=np.array,
            validator=validators.has_len(3),
        ),
        doc="A 3-vector specifying the up direction of the camera.\n"
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
                    "equisolid_full",
                )
            ),
        ),
        doc="Projection model, must be one of: equidistant | equisolid | "
        "stereographic | orthographic | polynomial. The ``polynomial`` model "
        "uses a calibrated *normalised* projection function "
        "``rho(theta) = lens_a * theta + lens_b * theta**2`` with "
        "``rho in [0, 1]``; in that mode ``fov`` is ignored, as the mapping is "
        "fully defined by the calibration coefficients. See ``lens_a`` and "
        "``lens_b`` for their role.",
        type="str",
        default="equisolid",
    )

    fov: pint.Quantity = documented(
        pinttrs.field(default=180.0 * ureg.deg, units=ureg.deg),
        doc="Camera field of view, i.e. the full angle subtended by the image "
        "circle. Must lie in (0, 360)°; the orthographic projection "
        "additionally requires at most 180°. Ignored when "
        "``projection_model == 'polynomial'``.\n"
        "\n"
        "Unit-enabled field (default: degree).",
        type="quantity",
        init_type="quantity or float",
        default="180°",
    )

    lens_a: float | None = documented(
        attrs.field(default=None, converter=attrs.converters.optional(float)),
        doc="Linear term of the polynomial projection formula (see "
        "``projection_model``). Required and must be strictly positive when "
        "``projection_model == 'polynomial'``; unused otherwise.",
        type="float or None",
        init_type="float, optional",
        default="None",
    )

    lens_b: float | None = documented(
        attrs.field(default=None, converter=attrs.converters.optional(float)),
        doc="Quadratic term of the polynomial projection formula (see "
        "``projection_model``). Defaults to 0 (purely linear projection) when "
        "unset. Used only when ``projection_model == 'polynomial'``.",
        type="float or None",
        init_type="float, optional",
        default="None",
    )

    max_radius: float | None = documented(
        attrs.field(default=None, converter=attrs.converters.optional(float)),
        doc="Calibrated image-circle radius ('Maximum Radius') in pixels at "
        "``calibration_resolution``, where the projection reaches ``rho = 1``. "
        "Kernel defaults to the inscribed circle if unset. Used only when "
        "``projection_model == 'polynomial'``.",
        type="float or None",
        init_type="float, optional",
        default="None",
    )

    center_x: float | None = documented(
        attrs.field(default=None, converter=attrs.converters.optional(float)),
        doc="Calibrated optical-centre x coordinate ('Lens X') in pixels at "
        "``calibration_resolution``. Kernel defaults to film centre if unset. "
        "Used only when ``projection_model == 'polynomial'``.",
        type="float or None",
        init_type="float, optional",
        default="None",
    )

    center_y: float | None = documented(
        attrs.field(default=None, converter=attrs.converters.optional(float)),
        doc="Calibrated optical-centre y coordinate ('Lens Y') in pixels at "
        "``calibration_resolution``. Kernel defaults to film centre if unset. "
        "Used only when ``projection_model == 'polynomial'``.",
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
                    member_validator=attrs.validators.instance_of(int),
                    iterable_validator=validators.has_len(2),
                )
            ),
        ),
        doc="Resolution (width, height) in which calibration data are "
        "expressed (pixel frame for ``center_x``, ``center_y``, ``max_radius``). "
        "Defaults to ``film_resolution`` if unset; aspect ratio must match. "
        "Used only when ``projection_model == 'polynomial'``.",
        type="tuple of int or None",
        init_type="array-like, optional",
        default="None",
    )

    far_clip: pint.Quantity = documented(
        pinttrs.field(
            default=1e4 * ureg.km,
            units=ucc.deferred("length"),
        ),
        doc="Distance to the far clip sphere: rays are traced up to this "
        "radial distance from the camera.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="quantity",
        init_type="quantity of float",
        default="10 000 km",
    )

    def update(self) -> None:
        # Inherit docstring
        super().update()

        fov = self.fov.m_as(ureg.deg)
        if not 0.0 < fov < 360.0:
            raise ValueError(f"fov must be in the (0, 360)° range, got {fov}°")
        if self.projection_model == "orthographic" and fov > 180.0:
            raise ValueError(
                f"the 'orthographic' projection model requires fov <= 180°, "
                f"got {fov}°"
            )

        calibration_fields = {
            "lens_a": self.lens_a,
            "lens_b": self.lens_b,
            "max_radius": self.max_radius,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "calibration_resolution": self.calibration_resolution,
        }

        if self.projection_model != "polynomial":
            set_fields = [k for k, v in calibration_fields.items() if v is not None]
            if set_fields:
                raise ValueError(
                    f"calibration fields {set_fields} are only used by the "
                    f"'polynomial' projection model, got "
                    f"projection_model = '{self.projection_model}'"
                )
            return

        if self.lens_a is None or self.lens_a <= 0:
            raise ValueError(
                f"the 'polynomial' projection model requires lens_a > 0, "
                f"got {self.lens_a}"
            )

        calib_w, calib_h = self.calibration_resolution or self.film_resolution
        film_w, film_h = self.film_resolution
        # Calibration sheets are hand-measured, so we allow small measurement error
        if not math.isclose(calib_w / calib_h, film_w / film_h, rel_tol=1e-3):
            raise ValueError(
                f"film_resolution {self.film_resolution} and "
                f"calibration_resolution {self.calibration_resolution} must "
                f"have the same aspect ratio"
            )
        if self.max_radius is not None and self.max_radius <= 0:
            raise ValueError(
                f"the 'polynomial' projection model requires max_radius > 0, "
                f"got {self.max_radius}"
            )
        if self.center_x is not None and not 0 <= self.center_x <= calib_w:
            raise ValueError(
                f"center_x must lie within the calibration frame "
                f"[0, {calib_w}], got {self.center_x}"
            )
        if self.center_y is not None and not 0 <= self.center_y <= calib_h:
            raise ValueError(
                f"center_y must lie within the calibration frame "
                f"[0, {calib_h}], got {self.center_y}"
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

        if self.projection_model == "polynomial":
            # Kernel expects normalised film coordinates; aspect-ratio
            # validation ensures the normalisation is consistent across
            # resolutions.
            calib_w, calib_h = self.calibration_resolution or self.film_resolution
            result["lens_a"] = self.lens_a
            result["lens_b"] = 0.0 if self.lens_b is None else self.lens_b
            if self.center_x is not None:
                result["center_x"] = self.center_x / calib_w
            if self.center_y is not None:
                result["center_y"] = self.center_y / calib_h
            if self.max_radius is not None:
                result["radius"] = self.max_radius / calib_w
        else:
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
