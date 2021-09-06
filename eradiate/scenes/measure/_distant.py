from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import attr
import numpy as np
import pint
import pinttr
import xarray as xr

import eradiate

from ._core import Measure, measure_factory
from ..illumination import ConstantIllumination, DirectionalIllumination
from ..spectra import Spectrum
from ... import converters, validators
from ..._mode import ModeFlags
from ..._util import is_vector3
from ...attrs import documented, parse_docs
from ...exceptions import UnsupportedModeError
from ...frame import direction_to_angles
from ...units import symbol, to_quantity
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg
from ...warp import square_to_uniform_hemisphere


@attr.s
class TargetOrigin:
    """
    Interface for target and origin selection classes used by :class:`DistantMeasure`.
    """

    def kernel_item(self) -> Dict:
        """Return kernel item."""
        raise NotImplementedError

    @staticmethod
    def new(target_type, *args, **kwargs) -> TargetOrigin:
        """
        Instantiate one of the supported child classes. This factory requires
        manual class registration. All position and keyword arguments are
        forwarded to the constructed type.

        Currently supported classes:

        * ``point``: :class:`TargetOriginPoint`
        * ``rectangle``: :class:`TargetOriginRectangle`
        * ``sphere``: :class:`TargetOriginSphere`

        Parameter ``target_type`` (str):
            Identifier of one of the supported child classes.
        """
        if target_type == "point":
            return TargetOriginPoint(*args, **kwargs)
        elif target_type == "rectangle":
            return TargetOriginRectangle(*args, **kwargs)
        elif target_type == "sphere":
            return TargetOriginSphere(*args, **kwargs)
        else:
            raise ValueError(f"unknown target type {target_type}")

    @staticmethod
    def convert(value) -> Any:
        """
        Object converter method.

        If ``value`` is a dictionary, this method uses :meth:`new` to
        instantiate a :class:`Target` child class based on the ``"type"`` entry
        it contains.

        If ``value`` is a 3-vector, this method returns a :class:`.TargetPoint`
        instance.

        Otherwise, it returns ``value``.
        """
        if isinstance(value, dict):
            d = deepcopy(value)
            try:
                target_type = d.pop("type")
            except KeyError:
                raise ValueError("cannot convert dict, missing 'type' entry")

            return TargetOrigin.new(target_type, **d)

        if is_vector3(value):
            return TargetOrigin.new("point", xyz=value)

        return value


def _target_point_rectangle_xyz_converter(x):
    return converters.on_quantity(float)(
        pinttr.converters.to_units(ucc.deferred("length"))(x)
    )


@parse_docs
@attr.s
class TargetOriginPoint(TargetOrigin):
    """
    Point target or origin specification.
    """

    # Target point in config units
    xyz: pint.Quantity = documented(
        pinttr.ib(units=ucc.deferred("length")),
        doc="Point coordinates.\n\nUnit-enabled field (default: ucc['length']).",
        type="array-like",
    )

    @xyz.validator
    def _xyz_validator(self, attribute, value):
        if not is_vector3(value):
            raise ValueError(
                f"while validating {attribute.name}: must be a "
                f"3-element vector of numbers"
            )

    def kernel_item(self) -> Dict:
        """Return kernel item."""
        return self.xyz.m_as(uck.get("length"))


@parse_docs
@attr.s
class TargetOriginRectangle(TargetOrigin):
    """
    Rectangle target origin specification.

    This class defines an axis-aligned rectangular zone where ray targets will
    be sampled or ray origins will be projected.
    """

    xmin: pint.Quantity = documented(
        pinttr.ib(
            converter=_target_point_rectangle_xyz_converter,
            units=ucc.deferred("length"),
        ),
        doc="Lower bound on the X axis.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="float",
    )

    xmax: pint.Quantity = documented(
        pinttr.ib(
            converter=_target_point_rectangle_xyz_converter,
            units=ucc.deferred("length"),
        ),
        doc="Upper bound on the X axis.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="float",
    )

    ymin: pint.Quantity = documented(
        pinttr.ib(
            converter=_target_point_rectangle_xyz_converter,
            units=ucc.deferred("length"),
        ),
        doc="Lower bound on the Y axis.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="float",
    )

    ymax: pint.Quantity = documented(
        pinttr.ib(
            converter=_target_point_rectangle_xyz_converter,
            units=ucc.deferred("length"),
        ),
        doc="Upper bound on the Y axis.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="float",
    )

    z: pint.Quantity = documented(
        pinttr.ib(
            default=0.0,
            converter=_target_point_rectangle_xyz_converter,
            units=ucc.deferred("length"),
        ),
        doc="Altitude of the plane enclosing the rectangle.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="float",
        default="0.0",
    )

    @xmin.validator
    @xmax.validator
    @ymin.validator
    @ymax.validator
    @z.validator
    def _xyz_validator(self, attribute, value):
        validators.on_quantity(validators.is_number)(self, attribute, value)

    @xmin.validator
    @xmax.validator
    def _x_validator(self, attribute, value):
        if not self.xmin < self.xmax:
            raise ValueError(
                f"while validating {attribute.name}: 'xmin' must "
                f"be lower than 'xmax"
            )

    @ymin.validator
    @ymax.validator
    def _y_validator(self, attribute, value):
        if not self.ymin < self.ymax:
            raise ValueError(
                f"while validating {attribute.name}: 'ymin' must "
                f"be lower than 'ymax"
            )

    def kernel_item(self):
        """Return kernel item."""
        from mitsuba.core import ScalarTransform4f

        xmin = self.xmin.m_as(uck.get("length"))
        xmax = self.xmax.m_as(uck.get("length"))
        ymin = self.ymin.m_as(uck.get("length"))
        ymax = self.ymax.m_as(uck.get("length"))
        z = self.z.m_as(uck.get("length"))

        dx = xmax - xmin
        dy = ymax - ymin

        to_world = ScalarTransform4f.translate(
            [0.5 * dx + xmin, 0.5 * dy + ymin, z]
        ) * ScalarTransform4f.scale([0.5 * dx, 0.5 * dy, 1.0])

        return {"type": "rectangle", "to_world": to_world}


@parse_docs
@attr.s
class TargetOriginSphere(TargetOrigin):
    """
    Sphere target or origin specification.
    """

    center: pint.Quantity = documented(
        pinttr.ib(units=ucc.deferred("length")),
        doc="Center coordinates.\n" "\n" "Unit-enabled field (default: ucc['length']).",
        type="array-like",
    )

    @center.validator
    def _center_validator(self, attribute, value):
        if not is_vector3(value):
            raise ValueError(
                f"while validating {attribute.name}: must be a "
                f"3-element vector of numbers"
            )

    radius: pint.Quantity = documented(
        pinttr.ib(
            units=ucc.deferred("length"),
            validator=[pinttr.validators.has_compatible_units, validators.is_positive],
        ),
        doc="Sphere radius.\n\nUnit-enabled field (default: ucc['length']).",
        type="float",
    )

    def kernel_item(self) -> Dict:
        """Return kernel item."""
        center = self.center.m_as(uck.get("length"))
        radius = self.radius.m_as(uck.get("length"))

        return {"type": "sphere", "center": center, "radius": radius}


@attr.s
class DistantMeasure(Measure):
    """
    Base class for all measures recording radiometric quantities at an infinite
    distance.
    """

    target: Optional[TargetOrigin] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(TargetOrigin.convert),
            validator=attr.validators.optional(
                attr.validators.instance_of(
                    (
                        TargetOriginPoint,
                        TargetOriginRectangle,
                    )
                )
            ),
            on_setattr=attr.setters.pipe(attr.setters.convert, attr.setters.validate),
        ),
        doc="Target specification. If set to ``None``, default target point "
        "selection is used: rays will not target a particular region of the "
        "scene. The target can be specified using an array-like with 3 "
        "elements (which will be converted to a :class:`.TargetOriginPoint`) "
        "or a dictionary interpreted by "
        ":meth:`TargetOrigin.convert() <.TargetOrigin.convert>`.",
        type=":class:`.TargetOrigin` or None",
        default="None",
    )

    origin: Optional[TargetOrigin] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(TargetOrigin.convert),
            validator=attr.validators.optional(
                attr.validators.instance_of((TargetOriginSphere,))
            ),
            on_setattr=attr.setters.pipe(attr.setters.convert, attr.setters.validate),
        ),
        doc="Ray origin specification. If set to ``None``, the default origin "
        "point selection strategy is used: ray origins will be projected to "
        "the scene's bounding sphere. Otherwise, ray origins are projected "
        "to the shape specified as origin. The origin can be specified using "
        "a dictionary interpreted by "
        ":meth:`TargetOrigin.convert() <.TargetOrigin.convert>`.",
        type=":class:`.TargetOriginSphere` or None",
        default="None",
    )

    def _postprocess_add_illumination(
        self,
        ds: xr.Dataset,
        illumination: Union[DirectionalIllumination, ConstantIllumination],
    ) -> xr.Dataset:
        """
        Processes a measure result dataset and add illumination data and m
        etadata to it. This function is to be used as part of the
        post-processing pipeline and is optional.

        Parameter ``ds`` (:class:`xarray.Dataset`):
            Result dataset.

        Parameter ``illumination`` (:class:`.DirectionalIllumination` or :class:`ConstantIllumination`):
            Illumination whose data is to be added to the result data set.

        Returns → :class:`xarray.Dataset`:
            Updated result dataset.

        Raises → TypeError:
            If ``illumination`` has an unsupported type.
        """
        k_irradiance_units = uck.get("irradiance")

        # Collect spectral coordinate values for verification purposes
        wavelengths_dataset = to_quantity(ds.w)

        def eval_illumination_spectrum(
            field_name: str, k_units: pint.Unit
        ) -> pint.Quantity:
            # Local helper function to help with illumination spectrum evaluation

            spectrum: Spectrum = getattr(illumination, field_name)

            if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
                # Very important: sort spectral coordinate
                wavelengths = np.sort(self.spectral_cfg.wavelengths)
                assert np.allclose(wavelengths, wavelengths_dataset)
                return spectrum.eval_mono(wavelengths).m_as(k_units)

            elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
                # Collect bins and wavelengths, evaluate spectrum
                bins = self.spectral_cfg.bins
                wavelengths = [bin.wcenter.m_as(ureg.nm) for bin in bins] * ureg.nm
                result = spectrum.eval_ckd(*bins).m_as(k_units)

                # Reorder data by ascending wavelengths
                indices = wavelengths.argsort()
                wavelengths = wavelengths[indices]
                assert np.allclose(wavelengths, wavelengths_dataset)
                result = result[indices]

                return result

            else:
                raise UnsupportedModeError(supported=("monochromatic", "ckd"))

        if isinstance(illumination, DirectionalIllumination):
            # Collect illumination angular data
            saa = illumination.azimuth.m_as(ureg.deg)
            sza = illumination.zenith.m_as(ureg.deg)
            cos_sza = np.cos(np.deg2rad(sza))

            # Add angular dimensions
            ds = ds.expand_dims({"sza": [sza], "saa": [saa]}, axis=(0, 1))
            ds.coords["sza"].attrs = {
                "standard_name": "solar_zenith_angle",
                "long_name": "solar zenith angle",
                "units": symbol("deg"),
            }
            ds.coords["saa"].attrs = {
                "standard_name": "solar_azimuth_angle",
                "long_name": "solar azimuth angle",
                "units": symbol("deg"),
            }

            # Collect illumination spectral data
            irradiances = eval_illumination_spectrum("irradiance", k_irradiance_units)

            # Add irradiance variable
            ds["irradiance"] = (
                ("sza", "saa", "w"),
                (irradiances * cos_sza).reshape((1, 1, len(irradiances))),
            )

        elif isinstance(illumination, ConstantIllumination):
            # Collect illumination spectral data
            k_radiance_units = uck.get("radiance")
            radiances = eval_illumination_spectrum("radiance", k_radiance_units)

            # Add irradiance variable
            ds["irradiance"] = (
                ("w",),
                np.pi * radiances.reshape((len(radiances),)),
            )

        else:
            raise TypeError(
                "keyword argument 'illumination' must be one of "
                "(DirectionalIllumination, ConstantIllumination), got a "
                f"{illumination.__class__.__name__}"
            )

        ds["irradiance"].attrs = {
            "standard_name": "horizontal_solar_irradiance_per_unit_wavelength",
            "long_name": "horizontal spectral irradiance",
            "units": symbol(k_irradiance_units),
        }

        return ds


@measure_factory.register(type_id="distant", allow_aliases=True)
@measure_factory.register(type_id="distant_radiance", allow_aliases=True)
@parse_docs
@attr.s
class DistantRadianceMeasure(DistantMeasure):
    """
    Record the radiance (in W/m²/sr(/nm)) leaving the scene at infinite distance.
    Depending on film resolution (*i.e.* storage discretisation), radiance is
    recorded for a single direction, in a plane or in an entire hemisphere.

    When used with a backward tracing algorithm, rays traced by the sensor
    target a shape which can be controlled through the ``target`` parameter.
    This feature is useful if one wants to compute the average radiance leaving
    a particular subset of the scene.

    .. note:: This scene element is a thin wrapper around the ``distant`` sensor
       kernel plugin.
    """

    _film_resolution: Tuple[int, int] = documented(
        attr.ib(
            default=(32, 32),
            validator=attr.validators.deep_iterable(
                member_validator=attr.validators.instance_of(int),
                iterable_validator=validators.has_len(2),
            ),
        ),
        doc="Film resolution as a (width, height) 2-tuple. "
        "If the height is set to 1, direction sampling will be restricted to a "
        "plane.",
        type="array-like",
        default="(32, 32)",
    )

    orientation: pint.Quantity = documented(
        pinttr.ib(
            default=ureg.Quantity(0.0, ureg.deg),
            validator=validators.is_positive,
            units=ucc.deferred("angle"),
        ),
        doc="Azimuth angle defining the orientation of the sensor in the "
        "horizontal plane.\n"
        "\n"
        "Unit-enabled field (default: ucc['angle']).",
        type="float",
        default="0.0 deg",
    )

    direction = documented(
        attr.ib(
            default=[0, 0, 1],
            converter=np.array,
            validator=validators.is_vector3,
        ),
        doc="A 3-vector orienting the hemisphere mapped by the measure.",
        type="array-like",
        default="[0, 0, 1]",
    )

    flip_directions = documented(
        attr.ib(default=None, converter=attr.converters.optional(bool)),
        doc=" If ``True``, sampled directions will be flipped.",
        type="bool",
        default="False",
    )

    @property
    def film_resolution(self):
        return self._film_resolution

    def _base_dicts(self) -> List[Dict]:
        result = []

        for sensor_info in self.sensor_infos():
            d = {
                "type": "distant",
                "id": sensor_info.id,
                "direction": self.direction,
                "orientation": [
                    np.cos(self.orientation.m_as(ureg.rad)),
                    np.sin(self.orientation.m_as(ureg.rad)),
                    0.0,
                ],
            }

            if self.target is not None:
                d["ray_target"] = self.target.kernel_item()

            if self.origin is not None:
                d["ray_origin"] = self.origin.kernel_item()

            if self.flip_directions is not None:
                d["flip_directions"] = self.flip_directions

            result.append(d)

        return result

    def postprocess(self) -> xr.Dataset:
        # Fetch results (SPP-split aggregated, CKD quadrature computed) as a Dataset
        result = self._postprocess_fetch_results()

        # Attach viewing angle coordinates
        result = self._postprocess_add_viewing_angles(result)

        return result

    def _postprocess_fetch_results(self) -> xr.Dataset:
        # Collect raw results, compute CKD quadrature, add appropriate metadata
        ds = self.results.to_dataset(aggregate_spps=True)

        if eradiate.mode().has_flags(ModeFlags.ANY_CKD):
            ds = self._postprocess_ckd_eval_quad(ds)

        # Rename raw field to outgoing radiance
        ds = ds.rename(raw="lo")
        ds["lo"].attrs = {
            "standard_name": "toa_outgoing_radiance_per_unit_wavelength",
            "long_name": "top-of-atmosphere outgoing spectral radiance",
            "units": symbol(uck.get("radiance")),
        }

        return ds

    def _postprocess_add_viewing_angles(self, ds: xr.Dataset) -> xr.Dataset:
        # Compute viewing angles at pixel locations
        # Angle computation must match the kernel plugin's direction sampling
        # routine
        xs = ds.coords["x"].data
        ys = ds.coords["y"].data
        if not np.allclose((len(xs), len(ys)), self.film_resolution):
            raise ValueError(
                f"raw data width and height ({len(xs)}, {len(ys)}) does not "
                f"match film size ({self.film_resolution})"
            )

        if self.film_resolution[1] == 1:  # Plane case
            theta = (90.0 - 180.0 * xs).reshape(1, len(xs))
            phi = np.full_like(theta, self.orientation.m_as("deg"))

        else:  # Hemisphere case
            xy = np.array([(x, y) for y in ys for x in xs])
            angles = direction_to_angles(square_to_uniform_hemisphere(xy)).m_as("deg")
            theta = angles[:, 0].reshape((len(xs), len(ys))).T
            phi = angles[:, 1].reshape((len(xs), len(ys))).T

        # Assign angles as non-dimension coords
        ds = ds.assign_coords(
            {
                "vza": (
                    ("y", "x"),
                    theta,
                    {
                        "standard_name": "viewing_zenith_angle",
                        "long_name": "viewing zenith angle",
                        "units": symbol("deg"),
                    },
                ),
                "vaa": (
                    ("y", "x"),
                    phi,
                    {
                        "standard_name": "viewing_azimuth_angle",
                        "long_name": "viewing azimuth angle",
                        "units": symbol("deg"),
                    },
                ),
            }
        )

        return ds


@measure_factory.register(type_id="distant_reflectance")
@parse_docs
@attr.s
class DistantReflectanceMeasure(DistantRadianceMeasure):
    """
    A specialised version of :class:`.DistantRadianceMeasure` with extra
    post-processing features to derive reflectance values from the recorded
    radiance.

    This measure produces meaningful results only with the
    :class:`.DirectionalIllumination` illumination model (the :meth:`postprocess`
    method will raise a ``TypeError`` if called with an incompatible illumination
    type).
    """

    def postprocess(self, illumination=None) -> xr.Dataset:
        """
        Return post-processed raw sensor results.

        Parameter ``illumination`` (:class:`.DirectionalIllumination`):
            Incoming radiance value. *This keyword argument is required.*

        Returns → :class:`~xarray.Dataset`:
            Post-processed results.

        Raises → TypeError:
            If ``illumination`` is missing or if it has an unsupported type.
        """
        if illumination is None:
            raise TypeError("missing required keyword argument 'illumination'")

        if not isinstance(illumination, DirectionalIllumination):
            TypeError(
                "keyword argument 'illumination' must be a "
                "DirectionalIllumination instance, got a "
                f"{illumination.__class__.__name__}"
            )

        # Get radiance data
        result = super(DistantReflectanceMeasure, self).postprocess()

        # Add illumination data
        result = self._postprocess_add_illumination(result, illumination)

        # Compute reflectance data
        result = self._postprocessing_add_reflectance(result)

        return result

    def _postprocessing_add_reflectance(self, ds: xr.Dataset) -> xr.Dataset:
        # Compute BRDF and BRF
        # We assume that all quantities are stored in kernel units
        ds["brdf"] = ds["lo"] / ds["irradiance"]
        ds["brdf"].attrs = {
            "standard_name": "brdf",
            "long_name": "bi-directional reflection distribution function",
            "units": symbol("1/sr"),
        }

        ds["brf"] = ds["brdf"] * np.pi
        ds["brf"].attrs = {
            "standard_name": "brf",
            "long_name": "bi-directional reflectance factor",
            "units": symbol("dimensionless"),
        }

        return ds


@measure_factory.register(type_id="distant_flux")
@parse_docs
@attr.s
class DistantFluxMeasure(DistantMeasure):
    """
    Record the exitant flux density (in W/m²(/nm)) at infinite distance in a
    hemisphere defined by its ``direction`` parameter.

    When used with a backward tracing algorithm, rays traced by the sensor
    target a shape which can be controlled through the ``target`` parameter.
    This feature is useful if one wants to compute the average flux leaving
    a particular subset of the scene.

    .. admonition:: Notes
       :class: note

       * Setting the ``target`` parameter is required to get meaningful results.
         Solver applications should take care of setting it appropriately.
       * The film resolution can be adjusted to manually stratify film sampling
         and reduce variance in results. The default 32x32 is generally a good
         choice, but scenes with sharp reflection lobes may benefit from higher
         values.
       * This scene element is a thin wrapper around the ``distantflux``
         sensor kernel plugin.
    """

    direction: np.ndarray = documented(
        attr.ib(
            default=[0, 0, 1],
            converter=np.array,
            validator=validators.is_vector3,
        ),
        doc="A 3-vector defining the normal to the reference surface for which "
        "the exitant flux density is computed.",
        type="array-like",
        default="[0, 0, 1]",
    )

    _film_resolution: Tuple[int, int] = documented(
        attr.ib(
            default=(32, 32),
            validator=attr.validators.deep_iterable(
                member_validator=attr.validators.instance_of(int),
                iterable_validator=validators.has_len(2),
            ),
        ),
        doc="Film resolution as a (width, height) 2-tuple.",
        type="array-like",
        default="(32, 32)",
    )

    @property
    def film_resolution(self) -> Tuple[int, int]:
        return self._film_resolution

    def _base_dicts(self) -> List[Dict]:
        from mitsuba.core import ScalarTransform4f, ScalarVector3f, coordinate_system

        result = []
        _, up = coordinate_system(self.direction)

        for sensor_info in self.sensor_infos():
            d = {
                "type": "distantflux",
                "id": sensor_info.id,
                "to_world": ScalarTransform4f.look_at(
                    origin=[0, 0, 0],
                    target=ScalarVector3f(self.direction),
                    up=up,
                ),
            }

            if self.target is not None:
                d["target"] = self.target.kernel_item()

            if self.origin is not None:
                d["origin"] = self.origin.kernel_item()

            result.append(d)

        return result

    def postprocess(self) -> xr.Dataset:
        # Fetch results (SPP-split aggregated) as a Dataset
        result = self._postprocess_fetch_results()

        return result

    def _postprocess_fetch_results(self) -> xr.Dataset:
        # Collect results and add appropriate metadata
        ds = super(DistantFluxMeasure, self).postprocess()

        # Add aggregate flux density field
        ds["flux"] = ds["raw"].sum(dim=("x", "y"))
        ds["flux"].attrs = {
            "standard_name": "toa_outgoing_flux_density_per_unit_wavelength",
            "long_name": "top-of-atmosphere outgoing spectral flux density",
            "units": symbol(uck.get("irradiance")),
        }

        return ds


@measure_factory.register(type_id="distant_albedo")
@parse_docs
@attr.s
class DistantAlbedoMeasure(DistantFluxMeasure):
    """
    A specialised version of the :class:`.DistantFluxMeasure` with extra
    post-processing features to derive albedo values from the recorded flux
    density.

    This measure produces meaningful results with both the
    :class:`.DirectionalIllumination` and :class:`.ConstantIllumination`
    illumination models.
    """

    def postprocess(
        self,
        illumination: Optional[
            Union[DirectionalIllumination, ConstantIllumination]
        ] = None,
    ) -> xr.Dataset:
        """
        Return post-processed raw sensor results.

        Parameter ``illumination`` (:class:`.DirectionalIllumination` or :class:`.ConstantIllumination`):
            Incoming radiance value. *This keyword argument is required.*

        Returns → :class:`~xarray.Dataset`:
            Post-processed results.

        Raises → TypeError:
            If ``illumination`` is missing or if it has an unsupported type.
        """
        if illumination is None:
            raise TypeError("missing required keyword argument 'illumination'")

        if not isinstance(
            illumination, (DirectionalIllumination, ConstantIllumination)
        ):
            raise TypeError(
                "keyword argument 'illumination' must be one of "
                "(DirectionalIllumination, ConstantIllumination), got a "
                f"{illumination.__class__.__name__}"
            )

        # Get radiance data
        result = super(DistantAlbedoMeasure, self).postprocess()

        # Add illumination data
        result = self._postprocess_add_illumination(result, illumination)

        # Compute albedo data
        result = self._postprocess_add_albedo(result)

        return result

    def _postprocess_add_albedo(self, ds: xr.Dataset) -> xr.Dataset:
        # Compute albedo
        # We assume that all quantities are stored in kernel units
        ds["albedo"] = ds["flux"] / ds["irradiance"]
        ds["albedo"].attrs = {
            "standard_name": "albedo",
            "long_name": "surface albedo",
            "units": "",
        }

        return ds
