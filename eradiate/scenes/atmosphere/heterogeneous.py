""" Heterogeneous atmosphere scene elements """
import tempfile
from pathlib import Path

import attr
import numpy as np
import xarray as xr

from .base import Atmosphere
from .radiative_properties.rad_profile import RadProfile, RadProfileFactory
from ..core import SceneElementFactory
from ...util.attrs import validator_is_file
from ...util.units import kernel_default_units as kdu


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

    This class builds a one-dimensional heterogeneous atmosphere. It expands as
    a ``heterogeneous`` kernel plugin, which takes as parameters a set of
    paths to volume data files. The radiative properties used to configure
    :class:`.HeterogeneousAtmosphere` can be specified two ways:

    - if the ``profile`` field is specified, kernel volume data files will be
      created using those data;
    - if the ``profile`` field is not specified (_i.e._ set to ``None``), kernel
      volume data files will be read from locations set in the ``albedo_fname``
      and ``sigma_t_fname`` attributes (which then must be set to paths
      pointing to existing files).

    If ``profile`` is specified:

    - if ``albedo_fname`` and ``sigma_t_fname`` are specified, data files will
      be written to those paths;
    - if ``albedo_fname`` and ``sigma_t_fname`` are not specified (_i.e._ set
      to ``None``), filenames will be generated based on ``cache_dir``.

    .. note::

       Generated files are not destroyed after execution and can be accessed
       using paths saved in the ``albedo_fname`` and ``sigma_t_fname`` fields.

    .. warning::

       While radiative properties specified using the ``profile`` field will be
       scaled according to kernel default unit override, existing volume data
       will not.

    .. rubric:: Constructor arguments / instance attributes

    ``profile`` (:class:`~eradiate.scenes.atmosphere.radiative_properties.rad_profile.RadProfile` or None):
        Radiative property profile used. If set, volume data files will be
        created from profile data to initialise the corresponding kernel
        plugin. If ``None``, :class:`.HeterogeneousAtmosphere` will assume
        that volume data files already exist.

    ``albedo_fname`` (path-like or None):
        Path to the single scattering albedo volume data file. If ``None``,
        a value will be created when the file will be requested.
        Default: ``None``.

    ``sigma_t_fname`` (path-like or None):
        Path to the extinction coefficient volume data file. If ``None``,
        a value will be created when the file will be requested.
        Default: ``None``.

    ``cache_dir`` (path-like or None):
        Path to a cache directory where volume data files will be created.
        If ``None``, a temporary cache directory will be used.
    """

    profile = attr.ib(
        default=None,
        converter=RadProfileFactory.convert,
        validator=attr.validators.optional(attr.validators.instance_of(RadProfile))
    )

    albedo_fname = attr.ib(
        default=None,
        converter=attr.converters.optional(Path)
    )

    @albedo_fname.validator
    def _albedo_fname_validator(self, attribute, value):
        # The file should exist if no albedo value is provided to create it
        if self.profile is None:
            if value is None:
                raise ValueError("if 'profile' is not set, "
                                 "'albedo_fname' must be set")
            try:
                return validator_is_file(self, attribute, value)
            except FileNotFoundError:
                raise

    sigma_t_fname = attr.ib(
        default=None,
        converter=attr.converters.optional(Path),
    )

    @sigma_t_fname.validator
    def _sigma_t_fname_validator(self, attribute, value):
        # The file should exist if no sigma_t value is provided to create it
        if self.profile is None:
            if value is None:
                raise ValueError("if 'profile' is not set, "
                                 "'sigma_t_fname' must be set")
            try:
                return validator_is_file(self, attribute, value)
            except FileNotFoundError:
                raise

    _cache_dir = attr.ib(
        default=None,
        converter=attr.converters.optional(Path)
    )

    _quantities = {
        "albedo": "albedo",
        "sigma_t": "collision_coefficient"
    }

    def __attrs_post_init__(self):
        # Prepare cache directory in case we'd need it
        if self._cache_dir is None:
            self._cache_dir = Path(tempfile.mkdtemp())
        else:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def make_volume_data(self, fields=None):
        """Create volume data files for requested fields.

        Parameter ``fields`` (str or list or None):
            If str, field for which to create volume data file. If list,
            fields for which to create volume data files. If ``None``,
            all supported fields are processed (``{"albedo", "sigma_t"}``).
            Default: ``None``.
        """
        supported_fields = set(self._quantities.keys())

        if fields is None:
            fields = supported_fields
        elif isinstance(fields, str):
            fields = {fields}

        if self.profile is None:
            raise ValueError("'profile' is not set, cannot write volume data "
                             "files")

        for field in fields:
            # Is the requested field supported?
            if field not in supported_fields:
                raise ValueError(f"field {field} cannot be used to create "
                                 f"volume data")

            # Does the considered field have values?
            field_quantity = getattr(self.profile, field)

            if field_quantity is None:
                raise ValueError(f"field {field} is empty, cannot create "
                                 f"volume data")

            # If file name is not specified, we create one
            field_fname = getattr(self, f"{field}_fname")
            if field_fname is None:
                field_fname = self._cache_dir / f"{field}.vol"
                setattr(self, f"{field}_fname", field_fname)

            # We have the data and the filename: we can create the file
            write_binary_grid3d(
                field_fname,
                field_quantity.to(kdu.get(self._quantities[field])).magnitude
            )

    @property
    def _width(self):
        """Return scene width based on configuration."""

        if self.width == "auto":  # Support for auto is currently disabled
            raise NotImplementedError
        else:
            return self.width

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
        if self.profile is not None:
            self.make_volume_data("albedo")
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
