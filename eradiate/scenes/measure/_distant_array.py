from __future__ import annotations

import typing as t
import warnings

import attr
import numpy as np
import pint
import xarray as xr

from ._core import SensorInfo, measure_factory
from ._distant import DistantMeasure
from ..illumination import DirectionalIllumination
from ...attrs import documented, parse_docs
from ...units import symbol
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


@measure_factory.register(type_id="distant_array")
@parse_docs
@attr.s
class DistantArrayMeasure(DistantMeasure):
    """
    Distant sensor array measure scene element [``distant_array``].

    This measure scene element is a thin wrapper around the ``DistantSensor``
    measures. It records the radiance along the directions given by the
    ``directions`` parameter.
    """

    directions: np.ndarray = documented(
        attr.ib(
            default=np.array([[0.0, 0.0, -1.0]]),
            converter=np.array,
        ),
        doc="A sequence of 3-vectors specifying distant sensor directions.",
        type="array-like",
        default="[[0, 0, -1]]",
    )

    @classmethod
    def from_angles(cls, angles: t.List[t.Tuple[float, float]], **kwargs):
        """
        Contruct a :class:`.DistantArrayMeasure` from directions specified in
        (zenith, azimuth) angle pairs.

        Parameters
        ----------
        angles : array-like
            Directions given by 2-vectors of zenith and azimuth angles
            in degrees.

        **kwargs : dict, optional
            Keyword arguments forwarded to the :class:`.DistantArrayMeasure`
            constructor.

        Returns
        -------
        :class:`.DistantArrayMeasure`
            Created measure object.
        """
        if isinstance(angles, pint.Quantity):
            directions_converted = angles.to(ureg.rad).magnitude
        else:
            directions_converted = np.deg2rad(angles)

        directions = [
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]
            for (theta, phi) in directions_converted
        ]
        return cls(directions=directions, **kwargs)

    @classmethod
    def azimuthal_ring(
        cls,
        zenith_angle: float,
        direction: t.Optional[t.List] = None,
        azimuth_resolution: float = 0,
        azimuth_steps: float = 0,
        **kwargs,
    ):
        """
        Construct a :class:`.DistantArrayMeasure`, covering a ring of constant
        zenith angle in the hemisphere given by the direction parameter.
        The angular resolution along the azimuthal ring can be defined by either
        the resolution or a number of steps. If both are given, the number of
        steps takes precedence and a warning is issued.

        Parameters
        ----------
        zenith_angle : float
            Zenith angle that defines the ring in the hemisphere.
            Zenith is measured relative to the ``direction`` parameter

            This parameter can be given as a float in degrees or a
            :class:`pint.Quantity`.

        direction : array-like
            Direction that defines the hemisphere in which the azimuthal ring is
            positioned. The ``zenith_angle`` parameter defines a constant angle
            with respect to this direction.

        azimuth_resolution : float
            Angular resolution along the azimuthal ring. Can be given as a
            :class:`pint.Quantity` or a float in degrees.

        azimuth_steps : float
            Number of steps along the azimuthal ring.

        **kwargs : dict, optional
            Keyword arguments forwarded to the :class:`.DistantArrayMeasure`
            constructor.

        Returns
        -------
        :class:`.DistantArrayMeasure`:
            Created measure object.
        """
        from mitsuba.core import Frame3f

        if direction is None:
            direction = [0, 0, 1]

        frame = Frame3f(direction)

        if isinstance(zenith_angle, pint.Quantity):
            theta = zenith_angle.to(ureg.rad).magnitude
        else:
            theta = np.deg2rad(zenith_angle)

        if azimuth_steps != 0:

            if azimuth_resolution != 0:
                warnings.warn(
                    "Azimuthal resolution and steps were both defined. "
                    "Azimuthal steps take precedence."
                )

            azimuth_angles = np.linspace(
                0, 2 * np.pi, num=azimuth_steps, endpoint=False
            )

        elif azimuth_resolution != 0:
            if isinstance(azimuth_resolution, pint.Quantity):
                azimuth_resolution = azimuth_resolution.to(ureg.deg).magnitude

            azimuth_angles = np.linspace(
                0, 2 * np.pi, num=int(360.0 / azimuth_resolution), endpoint=False
            )

        else:
            raise ValueError(
                "Neither azimuth resolution or number of steps were defined."
            )

        directions = [
            frame.to_local(
                np.array(
                    [
                        np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta),
                    ]
                )
            )
            for phi in azimuth_angles
        ]

        return cls(directions=directions, **kwargs)

    @directions.validator
    def _directions_validator(self, attribute, value):
        if value.shape[1] != 3:
            raise ValueError(
                f"While validating {attribute.name}: "
                f"Expected shape (N, 3), got {value.shape}"
            )

    def sensor_infos(self) -> t.List[SensorInfo]:
        spps = self._split_spp()
        sensor_count = len(self.directions)

        # determine order of magnitude for sensor count and pad the numerical
        # part of sensor_id accordingly
        string_padding = int(np.log10(sensor_count)) + 1
        ids = [
            f"{self.id}_ms" + f"{i}".zfill(string_padding) for i in range(sensor_count)
        ]

        if len(spps) == 1:
            return [SensorInfo(id=id, spp=spps[0]) for id in ids]

        else:
            return [
                SensorInfo(id=f"{id}_spp{i}", spp=spp)
                for id in ids
                for i, spp in enumerate(spps)
            ]

    def _postprocess_add_directions(self, ds: xr.Dataset) -> xr.Dataset:
        ds = ds.assign_coords(
            {"directions": ([str(direction) for direction in self.directions])}
        )

        return ds

    def _base_dicts(self) -> t.List[t.Dict]:
        from mitsuba.core import ScalarTransform4f, ScalarVector3f, coordinate_system

        result = []

        for sensor_info in self.sensor_infos():
            ms_id = sensor_info.id.split("_")[1]
            i = int(ms_id.lstrip("ms"))
            direction = self.directions[i]
            _, up = coordinate_system(direction)
            d = {
                "type": "distant",
                "id": sensor_info.id,
                "to_world": ScalarTransform4f.look_at(
                    origin=[0, 0, 0],
                    target=ScalarVector3f(direction),
                    up=up,
                ),
            }

            if self.target is not None:
                d["ray_target"] = self.target.kernel_item()

            if self.origin is not None:
                d["ray_origin"] = self.origin.kernel_item()

            result.append(d)

        return result

    @property
    def film_resolution(self) -> t.Tuple[int, int]:
        return (1, 1)

    def postprocess(self):
        # Collect raw results, add appropriate metadata
        ds = self.results.to_dataset(aggregate_spps=True)

        # Rename raw field to outgoing radiance
        ds = ds.rename(raw="lo")
        ds["lo"].attrs = {
            "standard_name": "toa_outgoing_radiance_per_unit_wavelength",
            "long_name": "top-of-atmosphere outgoing spectral radiance",
            "units": symbol(uck.get("radiance")),
        }

        ds = self._postprocess_add_directions(ds)

        return ds


@measure_factory.register(type_id="distant_array_reflectance")
@parse_docs
@attr.s
class DistantArrayReflectanceMeasure(DistantArrayMeasure):
    """
    Distant array reflectance measure scene element [``distant_array_reflectance``].

    This measure scene element is a thin wrapper around the ``DistantSensor``
    measures. It records either the radiance and reflectance along the directions given by the
    ``directions`` parameter.
    """

    def postprocess(self, illumination=None) -> xr.Dataset:
        """
        Return post-processed raw sensor results.

        Parameters
        ----------
        illumination : :class:`.DirectionalIllumination`
            Incoming radiance value. *This keyword argument is required.*

        Returns
        -------
        :class:`~xarray.Dataset`:
            Post-processed results.

        Raises
        ------
        TypeError
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

        # Collect raw results, add appropriate metadata
        ds = self.results.to_dataset(aggregate_spps=True)

        # Rename raw field to outgoing radiance
        ds = ds.rename(raw="lo")
        ds["lo"].attrs = {
            "standard_name": "toa_outgoing_radiance_per_unit_wavelength",
            "long_name": "top-of-atmosphere outgoing spectral radiance",
            "units": symbol(uck.get("radiance")),
        }

        # Add illumination data
        ds = self._postprocess_add_illumination(ds, illumination)

        # Compute reflectance data
        ds = self._postprocess_add_reflectance(ds)

        # Add directions
        ds = self._postprocess_add_directions(ds)

        return ds

    def _postprocess_add_reflectance(self, ds: xr.Dataset) -> xr.Dataset:
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
