from __future__ import annotations

import re
import typing as t
import warnings

import attr
import numpy as np
import pint
import pinttr
import xarray as xr
from pinttr.util import always_iterable, ensure_units

import eradiate

from ._core import Measure, SensorInfo
from ._distant import TargetOrigin, TargetOriginPoint, TargetOriginRectangle
from ._pipeline import Pipeline, PipelineStep
from ..core import KernelDict
from ... import validators
from ..._mode import ModeFlags
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext
from ...exceptions import UnsupportedModeError
from ...frame import angles_to_direction, direction_to_angles
from ...units import symbol
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg

# ------------------------------------------------------------------------------
#                             Local utilities
# ------------------------------------------------------------------------------

_is_sorted = lambda a: np.all(a[:-1] <= a[1:])


def _angles_in_plane(
    plane: float,
    theta: np.typing.ArrayLike,
    phi: np.typing.ArrayLike,
    raise_exc: bool = True,
):
    """
    Check that a set of (zenith, azimuth) pairs belong to a given hemisphere
    plane cut.

    Parameters
    ----------

    plane : float
        Plane cut orientation in degrees.

    theta : ndarray
        List of zenith angle values with (N,) shape in degrees.

    phi : ndarray
        List of azimuth angle values with (N,) shape.

    raise_exc : bool, optional
        If ``True``, raise if not all directions are snapped to the specified
        hemisphere plane cut.

    Returns
    -------
    in_plane_positive, in_plane_negative
        Masks indicating indexes of directions contained in the positive (resp.
        negative) half-plane.

    Raises
    ------
    ValueError
        If not all directions are snapped to the specified hemisphere plane cut.
    """
    in_plane_positive = np.isclose(plane, phi) | np.isclose(0.0, theta)
    in_plane_negative = np.isclose((plane + 180) % 360, phi) & ~in_plane_positive
    in_plane = in_plane_positive | in_plane_negative

    if raise_exc and not (np.all(in_plane)):
        raise ValueError(
            "This step was configured to map plane cut data, but "
            "the input data contains off-plane points"
        )

    return in_plane_positive, in_plane_negative


@ureg.wraps(ret=("deg", "deg"), args=("deg", "deg", "deg"), strict=True)
def _remap_viewing_angles_plane(
    plane: np.typing.ArrayLike,
    theta: np.typing.ArrayLike,
    phi: np.typing.ArrayLike,
) -> t.Tuple[np.typing.ArrayLike, np.typing.ArrayLike]:
    r"""
    Remap viewing angles to a hemispherical plane cut.

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
    plane = plane % 360
    theta = theta % 360
    phi = phi % 360

    # Check that phi values are compatible with requested plane
    in_plane_positive, in_plane_negative = _angles_in_plane(plane, theta, phi)

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


# ------------------------------------------------------------------------------
#                       Post-processing pipeline steps
# ------------------------------------------------------------------------------


def _spectral_dims():
    if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
        return (
            (
                "w",
                {
                    "standard_name": "wavelength",
                    "long_name": "wavelength",
                    "units": ucc.get("wavelength"),
                },
            ),
        )
    elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
        return (
            ("bin", {"standard_name": "ckd_name", "long_name": "CKD bin"}),
            ("index", {"standard_name": "ckd_index", "long_name": "CKD index"}),
        )
    else:
        raise UnsupportedModeError


@parse_docs
@attr.s
class Assemble(PipelineStep):
    """
    Assemble raw kernel results (output as nested dictionaries) into an xarray
    dataset.

    This pipeline step takes a nested dictionary produced by the spectral loop
    of an :class:`.Experiment` and repackages it as a :class:`~xarray.Dataset`.
    The top-level spectral index is mapped to mode-dependent spectral
    coordinates. Results from the sensors part of a single measurement are
    mapped to specified sensor dimensions, which can then be further processed
    by aggregation steps. Film dimensions are left unmodified and retain their
    metadata.

    An ``img`` variable holds sensor values. An ``spp`` variable holds sample
    count.
    """

    prefix: str = documented(
        attr.ib(default=r".*"),
        default="r'.*'",
        type="str",
        init_type="str, optional",
        doc="Prefix string used to match sensor IDs. The default value will "
        "match anything.",
    )

    sensor_dims: t.Sequence = documented(
        attr.ib(
            factory=list,
            validator=attr.validators.instance_of((list, tuple)),
        ),
        default="[]",
        type="list of (str or tuple)",
        init_type="list of (str or tuple), optional",
        doc="List of sensor dimensions. Each list item can be a string or a "
        "(dimension, metadata) pair.",
    )

    img_var: t.Union[str, t.Tuple[str, t.Dict]] = documented(
        attr.ib(default="img"),
        default="img",
        type="str or tuple[str, dict]",
        init_type="str or tuple[str, dict], optional",
        doc="Name of the variable containing sensor data. Optionally, a "
        "(name, metadata) pair can be passed.",
    )

    def transform(self, x: t.Dict) -> xr.Dataset:
        # Basic preparation
        prefix = self.prefix

        sensor_dims = []
        sensor_dim_metadata = {}

        for y in self.sensor_dims:
            if isinstance(y, str):
                sensor_dims.append(y)
                sensor_dim_metadata[y] = {}
            else:
                sensor_dims.append(y[0])
                sensor_dim_metadata[y[0]] = y[1]

        spectral_dims = []
        spectral_dim_metadata = {}

        for y in _spectral_dims():
            if isinstance(y, str):
                spectral_dims.append(y)
                spectral_dim_metadata[y] = {}
            else:
                spectral_dims.append(y[0])
                spectral_dim_metadata[y[0]] = y[1]

        sensor_datasets = []

        regex = re.compile(
            r"\_".join(
                [prefix]
                + [rf"{sensor_dim}(?P<{sensor_dim}>\d*)" for sensor_dim in sensor_dims]
            )
        )

        # Loop on spectral indexes
        for spectral_index in x.keys():
            # Loop on sensors
            for sensor_id, sensor_data in x[spectral_index]["values"].items():
                # Collect data
                ds = sensor_data.copy(deep=False)
                spp = x[spectral_index]["spp"][sensor_id]

                # Set spectral coordinates
                spectral_coords = {
                    spectral_dim: [spectral_coord]
                    for spectral_dim, spectral_coord in zip(
                        spectral_dims, always_iterable(spectral_index)
                    )
                }

                # Detect sensor coordinates
                match = regex.match(sensor_id)

                if match is None:
                    raise RuntimeError(
                        "could not detect requested sensor dimensions in "
                        f"sensor ID '{sensor_id}' using regex '{regex.pattern}'; "
                        "this could be due to incorrect values or order of "
                        "'sensor_dims'"
                    )

                sensor_coords = {
                    f"{sensor_dim}_index": [int(match.group(sensor_dim))]
                    for sensor_dim in sensor_dims
                }

                # Add spp dimension even though sample count split did not
                # produce any extra sensor
                if "spp" not in sensor_dims:
                    spectral_coords["spp_index"] = [0]

                # Add spectral and sensor dimensions to img array
                ds["img"] = ds.img.expand_dims(dim={**spectral_coords, **sensor_coords})

                # Package spp in a data array
                ds["spp"] = ("spp_index", [spp])

                sensor_datasets.append(ds)

        # Combine all the data
        with xr.set_options(keep_attrs=True):
            result = xr.merge(sensor_datasets)

        # Apply metadata to new dimensions
        for sensor_dim in sensor_dims:
            result[f"{sensor_dim}_index"].attrs = sensor_dim_metadata[sensor_dim]

        if "spp" not in sensor_dims:
            result["spp_index"].attrs = {
                "standard_name": "spp_index",
                "long_name": "SPP index",
            }

        for spectral_dim in spectral_dims:
            result[spectral_dim].attrs = spectral_dim_metadata[spectral_dim]

        # Apply metadata to data variables
        if isinstance(self.img_var, str):
            img_var = self.img_var
            img_var_metadata = {}
        else:
            img_var = self.img_var[0]
            img_var_metadata = self.img_var[1]

        result = result.rename({"img": img_var})
        result[img_var].attrs.update(img_var_metadata)

        result["spp"].attrs = {
            "standard_name": "sample_count",
            "long_name": "sample count",
        }

        return result


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


@parse_docs
@attr.s
class MultiDistantMeasure(Measure):

    # --------------------------------------------------------------------------
    #                  Specific post-processing pipeline steps
    # --------------------------------------------------------------------------

    @attr.s
    class AddViewingAngles(PipelineStep):
        """
        Create new ``vza`` and ``vaa`` coordinate variables mapping viewing angles
        to other coordinates.
        """

        multi_distant: MultiDistantMeasure = documented(
            attr.ib(
                repr=lambda self: f"{self.__class__.__name__}(id='{self.id}', ...)",
            ),
            doc="A :class:`.MultiDistantMeasure` instance from which the "
            "processed data originates.",
            type=":class:`.MultiDistantMeasure`",
        )

        @multi_distant.validator
        def _multi_distant_validator(self, attribute, value):
            # We must use a decorated validator definition because
            # MultiDistantMeasure isn't defined in the class definition scope
            attr.validators.instance_of(MultiDistantMeasure)(self, attribute, value)

        plane: t.Optional[pint.Quantity] = documented(
            pinttr.ib(default=None, units=ucc.deferred("angle")),
            doc="If set, indicates that the angles are to be mapped to a "
            "hemisphere plane cut. The value then defines the orientation of "
            "the plane cut. Directions with an azimuth equal to `plane` are "
            "mapped to positive zenith values; directions with an azimuth "
            "equal to `plane` + 180° are mapped to negative zenith values.",
            type="quantity or None",
            init_type="quantity, optional",
            default="None",
        )

        def transform(self, x: t.Any) -> t.Any:
            viewing_angles = self.multi_distant.viewing_angles

            # Collect zenith and azimuth values
            theta = viewing_angles[:, 0]
            phi = viewing_angles[:, 1]

            if self.plane is not None:
                theta, phi = _remap_viewing_angles_plane(self.plane, theta, phi)

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

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    split_spp: t.Optional[int] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(int),
            validator=attr.validators.optional(validators.is_positive),
        ),
        type="int",
        init_type="int, optional",
        doc="If set, this measure will be split into multiple sensors, each "
        "with a sample count lower or equal to `split_spp`. This parameter "
        "should be used in single-precision modes when the sample count is "
        "higher than 100,000 (very high sample count might result in floating "
        "point number precision issues otherwise).",
    )

    @split_spp.validator
    def _split_spp_validator(self, attribute, value):
        if (
            eradiate.mode().has_flags(ModeFlags.ANY_SINGLE)
            and self.spp > 1e5
            and self.split_spp is None
        ):
            warnings.warn(
                "In single-precision modes, setting a sample count ('spp') to "
                "values greater than 100,000 may result in floating point "
                "precision issues: using the measure's 'split_spp' parameter is "
                "recommended."
            )

    directions: np.ndarray = documented(
        attr.ib(
            default=np.array([[0.0, 0.0, -1.0]]),
            converter=np.array,
        ),
        doc="A sequence of 3-vectors specifying distant sensing directions.",
        type="ndarray",
        init_type="array-like",
        default="[[0, 0, -1]]",
    )

    target: t.Optional[TargetOrigin] = documented(
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
        doc="Target specification. The target can be specified using an "
        "array-like with 3 elements (which will be converted to a "
        ":class:`.TargetOriginPoint`) or a dictionary interpreted by "
        ":meth:`TargetOrigin.convert() <.TargetOrigin.convert>`. If set to "
        "``None`` (not recommended), the default target point selection "
        "method is used: rays will not target a particular region of the "
        "scene.",
        type=":class:`.TargetOrigin` or None",
        init_type=":class:`.TargetOrigin` or dict or array-like, optional",
    )

    post_processing_pipeline: Pipeline = documented(
        attr.ib(
            converter=Pipeline.convert,
            validator=attr.validators.instance_of(Pipeline),
        ),
        doc="Post-processing pipeline. This parameter defaults to a basic "
        "pipeline suitable for most commonly encountered situations.",
        type=":class:`.Pipeline`",
        init_type=":class:`.Pipeline` or list of tuple, optional",
    )

    @post_processing_pipeline.default
    def default_pipeline(self):
        """
        Generate a default post-processing pipeline. It includes the following
        steps:

        1. :class:`.Assemble`
        2. :class:`.AggregateSampleCount`
        3. :class:`.AddViewingAngles`

        Returns
        -------
        :class:`.Pipeline`
        """
        assemble = Assemble(
            sensor_dims=["spp"] if self.split_spp else [],
            img_var=(
                "lo",
                {
                    "units": uck.get("radiance"),
                    "standard_name": "leaving_radiance",
                    "long_name": "leaving radiance",
                },
            ),
        )
        add_viewing_angles = self.AddViewingAngles(multi_distant=self)

        return Pipeline(
            [
                ("assemble", assemble),
                ("aggregate_sample_count", AggregateSampleCount()),
                ("add_viewing_angles", add_viewing_angles),
                # ("add_illumination", AddIllumination()),
                # ("compute_reflectance", ComputeReflectance()),
            ]
        )

    @property
    def viewing_angles(self) -> pint.Quantity:
        """
        quantity: Viewing angles computed from stored `directions` as a (N, 2)
            array, where N is the number of directions. The second dimension
            is ordered as (zenith, azimuth).
        """
        return direction_to_angles(-self.directions).to(ucc.get("angle"))

    @property
    def film_resolution(self) -> t.Tuple[int, int]:
        return (self.directions.shape[0], 1)

    # --------------------------------------------------------------------------
    #                         Additional constructors
    # --------------------------------------------------------------------------

    @classmethod
    def from_viewing_angles(
        cls,
        angles: np.typing.ArrayLike,
        plane: t.Optional[np.typing.ArrayLike] = None,
        **kwargs,
    ):
        """
        Construct a :class:`.MultiDistantMeasure` using viewing angles instead
        of raw directions.

        Parameters
        ----------
        angles : array-like
            A list of (zenith, azimuth) pairs, possibly wrapped in a
            :class:`~pint.Quantity`. Unitless values are automatically converted
            to configuration units by the configuration unit context
            (:data:`.unit_config_context`).

        plane : float or quantity
            If all directions are expected to be within a hemisphere plane cut,
            the azimuth value of that plane. Unitless values are automatically
            converted to configuration units by the configuration unit context
            (:data:`.unit_config_context`).

        **kwargs
            Any keyword argument (except `direction`) to be forwarded to
            :class:`MultiDistantMeasure() <.MultiDistantMeasure>`.

        Returns
        -------
        MultiDistantMeasure
        """
        if "directions" in kwargs:
            raise TypeError(
                "from_viewing_angles() got an unexpected keyword argument 'directions'"
            )

        angles = pinttr.util.ensure_units(angles, default_units=ucc.get("angle"))
        directions = -angles_to_direction(angles)
        result = cls(directions=directions, **kwargs)

        if plane is not None:
            plane = ensure_units(plane, default_units=ucc.get("angle"))
            # Check that all specified directions are in the requested plane
            _angles_in_plane(
                plane.m_as(ureg.deg),
                angles[:, 0].m_as(ureg.deg),
                angles[:, 1].m_as(ureg.deg),
                raise_exc=True,
            )
            # Update post-processing pipeline
            result.post_processing_pipeline.plane = plane

        return result

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    def _sensor_spps(self) -> t.List[int]:
        if self.split_spp is not None and self.spp > self._spp_splitting_threshold:
            spps = [self.split_spp] * int(self.spp / self.split_spp)

            if self.spp % self.split_spp:
                spps.append(self.spp % self.split_spp)

            return spps

        else:
            return [self.spp]

    def _sensor_id(self, i_spp=None):
        """
        Assemble a sensor ID from indexes on sensor coordinates.
        """
        components = [self.id]

        if i_spp is not None:
            components.append(f"spp{i_spp}")

        return "_".join(components)

    def _sensor_ids(self) -> t.List[str]:
        if self.split_spp is not None and self.spp > self._spp_splitting_threshold:
            return [self._sensor_id(i) for i, _ in enumerate(self._sensor_spps())]

        else:
            return [self._sensor_id()]

    def _kernel_dict(self, sensor_id, spp):
        result = {
            "type": "mdistant",
            "id": sensor_id,
            "directions": ",".join(map(str, self.directions.ravel(order="C"))),
            "sampler": {
                "type": "independent",
                "sample_count": spp,
            },
            "film": {
                "type": "hdrfilm",
                "width": self.film_resolution[0],
                "height": self.film_resolution[1],
                "pixel_format": "luminance",
                "component_format": "float32",
                "rfilter": {"type": "box"},
            },
        }

        if self.target is not None:
            result["target"] = self.target.kernel_item()

        return result

    def sensor_infos(self) -> t.List[SensorInfo]:
        return [
            SensorInfo(id=id, spp=spp)
            for id, spp in zip(self._sensor_ids(), self._sensor_spps())
        ]

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        sensor_ids, sensor_spps = [], []

        for x in self.sensor_infos():
            sensor_ids.append(x.id)
            sensor_spps.append(x.spp)

        result = KernelDict()

        for spp, sensor_id in zip(sensor_spps, sensor_ids):
            result.data[sensor_id] = self._kernel_dict(sensor_id, spp)

        return result

    def _base_dicts(self) -> t.List[t.Dict]:
        raise NotImplementedError
