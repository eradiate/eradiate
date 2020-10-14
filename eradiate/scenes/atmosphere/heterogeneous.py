""" Heterogeneous atmosphere scene elements """
import tempfile
from pathlib import Path

import attr
import numpy as np
import xarray as xr

from .base import Atmosphere
from ..core import SceneElementFactory
from ...util.attrs import attrib, attrib_units, validator_is_file
from ...util.units import config_default_units as cdu
from ...util.units import kernel_default_units as kdu
from ...util.units import ureg


def write_binary_grid3d(filename, values):
    """Writes volume data to a binary file so that a ``gridvolume`` kernel
    plugin can be instantiated with that file.

    Parameter ``filename`` (path-like):
        File name.

    Parameter ``values`` (:class:`~numpy.ndarray` or :class:`~xarray.DataArray`):
        Data array to output to the volume data file. This array must 3 or 4
        dimensions (x, y, z, spectrum). If the array is 3-dimensional, it will
        automatically be assumed to have only one spectral channel.
    """
    if isinstance(values, xr.DataArray):
        values = values.values

    if not isinstance(values, np.ndarray):
        raise TypeError(f"unsupported data type {type(values)} "
                        f"(expected numpy array or xarray DataArray)")

    # note: this is an exact copy of the function write_binary_grid3d from
    # https://github.com/mitsuba-renderer/mitsuba-data/blob/master/tests/scenes/participating_media/create_volume_data.py

    with open(filename, 'wb') as f:
        f.write(b'V')
        f.write(b'O')
        f.write(b'L')
        f.write(np.uint8(3).tobytes())  # Version
        f.write(np.int32(1).tobytes())  # type
        f.write(np.int32(values.shape[0]).tobytes())  # size
        f.write(np.int32(values.shape[1]).tobytes())
        f.write(np.int32(values.shape[2]).tobytes())
        if values.ndim == 3:
            f.write(np.int32(1).tobytes())  # channels
        else:
            f.write(np.int32(values.shape[3]).tobytes())  # channels
        f.write(np.float32(0.0).tobytes())  # bbox
        f.write(np.float32(0.0).tobytes())
        f.write(np.float32(0.0).tobytes())
        f.write(np.float32(1.0).tobytes())
        f.write(np.float32(1.0).tobytes())
        f.write(np.float32(1.0).tobytes())
        f.write(values.ravel().astype(np.float32).tobytes())


def _dataarray_to_ndarray(value):
    if isinstance(value, xr.DataArray):
        return value.values
    else:
        return value


@SceneElementFactory.register("heterogeneous")
@attr.s
class HeterogeneousAtmosphere(Atmosphere):
    r"""Heterogeneous atmosphere scene element
    [:factorykey:`heterogeneous`].

    See :class:`~eradiate.scenes.atmosphere.base.Atmosphere` for undocumented
    members.

    This class builds a one-dimensional heterogeneous atmosphere. The used
    radiative properties can be specified two ways:

    - if the ``albedo`` and ``sigma_t`` fields are specified, kernel volume data
      files will be created using those data;
    - if the ``albedo`` and ``sigma_t`` fields are not specified (_i.e._ set to
      ``None``), kernel volume data files will be read from locations set in
      the ``albedo_fname`` and ``sigma_t_fname`` attributes.

    .. note::

       It is possible to mix and match approaches (_e.g._ provide an array
       for ``albedo`` and a file path for ``sigma_t``.

    If ``albedo`` and ``sigma_t`` are specified:

    - if ``albedo_fname`` and ``sigma_t_fname`` are specified, data files will
      be written to those paths;
    - if ``albedo_fname`` and ``sigma_t_fname`` are not specified (_i.e._ set
      to ``None``), filenames will be generated based on ``cache_dir``.

    .. note::

       Generated files are not destroyed after execution and can be accessed
       using paths saved in the ``albedo_fname`` and ``sigma_t_fname`` fields.

    .. warning::

       While radiative properties specified using the ``albedo`` and ``sigma_t``
       fields will be scaled according by default unit override, existing volume
       data will not.

    Constructor arguments / instance attributes:
        ``albedo`` (:class:`~numpy.ndarray` or :class:`~xarray.DataArray` or None):
            Array containing albedo values. If ``None``, volume data will be
            directly loaded from ``albedo_fname``.

            Unit-enabled field (default unit: dimensionless).

        ``albedo_fname`` (path-like or None):
            Path to the single scattering albedo volume data file. If ``None``,
            a value will be created when the file will be requested.
            Default: ``None``.

        ``sigma_t`` (:class:`~numpy.ndarray` or :class:`~xarray.DataArray` or None):
            Array containing scattering coefficient values. If ``None``, volume
            data will be directly loaded from ``sigma_t_fname``.

            Unit-enabled field (default unit: cdu[length]^-1).

        ``sigma_t_fname`` (path-like or None):
            Path to the extinction coefficient volume data file. If ``None``,
            a value will be created when the file will be requested.
            Default: ``None``.

        ``cache_dir`` (path-like or None):
            Path to a cache directory where volume data files will be created.
            If ``None``, a temporary cache directory will be used.
    """

    albedo = attrib(
        default=None,
        converter=_dataarray_to_ndarray,
        validator=attr.validators.optional(attr.validators.instance_of(np.ndarray)),
        has_units=True
    )

    albedo_units = attrib_units(
        compatible_units=ureg.dimensionless,
        default=attr.Factory(lambda: cdu.get("dimensionless"))
    )

    albedo_fname = attrib(
        default=None,
        converter=attr.converters.optional(Path)
    )

    @albedo_fname.validator
    def _albedo_fname_validator(self, attribute, value):
        # The file should exist if no albedo value is provided to create it
        if self.albedo is None:
            if value is None:
                raise ValueError("if 'albedo' is not set, "
                                 "'albedo_fname' must be set")
            try:
                return validator_is_file(self, attribute, value)
            except FileNotFoundError:
                raise

    @property
    def _albedo_quantity(self):
        """Returns → ``"albedo"``"""
        return "albedo"

    sigma_t = attrib(
        default=None,
        converter=_dataarray_to_ndarray,
        validator=attr.validators.optional(attr.validators.instance_of(np.ndarray)),
        has_units=True
    )

    sigma_t_units = attrib_units(
        compatible_units=ureg.m ** -1,
        default=attr.Factory(lambda: cdu.get("length") ** -1)
    )

    sigma_t_fname = attrib(
        default=None,
        converter=attr.converters.optional(Path),
    )

    @sigma_t_fname.validator
    def _sigma_t_fname_validator(self, attribute, value):
        # The file should exist if no sigma_t value is provided to create it
        if self.sigma_t is None:
            if value is None:
                raise ValueError("if 'sigma_t' is not set, "
                                 "'sigma_t_fname' must be set")
            try:
                return validator_is_file(self, attribute, value)
            except FileNotFoundError:
                raise

    @property
    def _sigma_t_quantity(self):
        """Returns → ``"collision_coefficient"``"""
        return "collision_coefficient"

    _cache_dir = attrib(
        default=None,
        converter=attr.converters.optional(Path)
    )

    def __attrs_post_init__(self):
        super(HeterogeneousAtmosphere, self).__attrs_post_init__()

        # Prepare cache directory in case we'd need it
        if self._cache_dir is None:
            self._cache_dir = Path(tempfile.mkdtemp())
        else:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def make_volume_data(self, fields=None):
        supported_fields = {"albedo", "sigma_t"}
        if fields is None:
            fields = supported_fields
        elif isinstance(fields, str):
            fields = {fields}

        for field in fields:
            # Is the requested field supported?
            if field not in supported_fields:
                raise ValueError(f"field {field} cannot be used to create "
                                 f"volume data")

            # Does the considered field have values?
            field_values = getattr(self, field)
            if field_values is None:
                raise ValueError(f"field {field} is empty, cannot create "
                                 f"volume data")
            field_units = getattr(self, f"{field}_units")
            field_quantity = ureg.Quantity(field_values, field_units)

            # If file name is not specified, we create one
            field_fname = getattr(self, f"{field}_fname")
            if field_fname is None:
                field_fname = self._cache_dir / f"{field}.vol"
                setattr(self, f"{field}_fname", field_fname)

            # We have the data and the filename: we can create the file
            write_binary_grid3d(
                field_fname,
                field_quantity.to(kdu.get(getattr(self, f"_{field}_quantity"))).magnitude
            )

    @property
    def _width(self):
        """Return scene width based on configuration."""

        if self.width == "auto":  # Support for auto is currently disabled
            raise NotImplementedError
        else:
            return self.get_quantity("width")

    def phase(self):
        return {f"phase_{self.id}": {"type": "rayleigh"}}

    def media(self, ref=False):
        from eradiate.kernel.core import ScalarTransform4f

        width = self._width.to(kdu.get("length")).magnitude
        height, offset = self._height
        height = height.to(kdu.get("length")).magnitude
        offset = offset.to(kdu.get("length")).magnitude

        # First, transform the [0, 1]^3 cube to the right dimensions
        trafo = ScalarTransform4f([
            [width, 0., 0., -0.5 * width],
            [0., width, 0., -0.5 * width],
            [0., 0., height + offset, -offset],
            [0., 0., 0., 1.],
        ])

        # Create volume data files if possible
        if self.albedo is not None:
            self.make_volume_data("albedo")

        if self.sigma_t is not None:
            self.make_volume_data("sigma_t")

        # Output kernel dict
        return {
            f"medium_{self.id}": {
                "type": "heterogeneous",
                "phase": {"type": "rayleigh"},
                "sigma_t": {
                    "type": "gridvolume",
                    "filename": str(self.sigma_t_fname),
                    "to_world": trafo
                },
                "albedo": {
                    "type": "gridvolume",
                    "filename": str(self.albedo_fname),
                    "to_world": trafo
                },
            }
        }

    def shapes(self, ref=False):
        from eradiate.kernel.core import ScalarTransform4f

        if ref:
            medium = {"type": "ref", "id": f"medium_{self.id}"}
        else:
            medium = self.media(ref=False)[f"medium_{self.id}"]

        width = self._width.to(kdu.get("length")).magnitude
        height, offset = self._height
        height = height.to(kdu.get("length")).magnitude
        offset = offset.to(kdu.get("length")).magnitude

        return {
            f"shape_{self.id}": {
                "type":
                    "cube",
                "to_world":
                    ScalarTransform4f([
                        [0.5 * width, 0., 0., 0.],
                        [0., 0.5 * width, 0., 0.],
                        [0., 0., 0.5 * (height + offset), 0.5 * (height - offset)],
                        [0., 0., 0., 1.],
                    ]),
                "bsdf": {
                    "type": "null"
                },
                "interior":
                    medium
            }
        }
