""" Heterogeneous atmosphere scene elements """
import struct
import tempfile
from pathlib import Path

import attr
import numpy as np
import xarray as xr

from ._base import (
    Atmosphere,
    AtmosphereFactory
)
from ..._attrs import (
    documented,
    parse_docs
)
from ..._units import unit_context_kernel as uck
from ..._units import unit_registry as ureg
from ...radprops import RadProfileFactory
from ...radprops.rad_profile import RadProfile
from ...validators import is_file


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

    if values.ndim not in {3, 4}:
        raise ValueError(f"'values' must have 3 or 4 dimensions "
                         f"(got shape {values.shape})")

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


def read_binary_grid3d(filename):
    """Reads a volume data binary file.

    Parameter ``filename`` (str):
        File name.

    Returns → :class:`~numpy.ndarray`:
        Values.
    """

    with open(filename, 'rb') as f:
        file_content = f.read()
        _shape = struct.unpack("iii", file_content[8:20])  # shape of the values array
        _num = np.prod(np.array(_shape))  # number of values
        values = np.array(struct.unpack("f" * _num, file_content[48:]))
        # file_type = struct.unpack("ccc", file_content[:3]),
        # version = struct.unpack("B", file_content[3:4]),
        # type = struct.unpack("i", file_content[4:8]),
        # channels = struct.unpack("i", file_content[20:24]),
        # bbox = struct.unpack("ffffff", file_content[24:48]),

    return values


def _dataarray_to_ndarray(value):
    if isinstance(value, xr.DataArray):
        return value.values
    else:
        return value


@AtmosphereFactory.register("heterogeneous")
@parse_docs
@attr.s
class HeterogeneousAtmosphere(Atmosphere):
    r"""Heterogeneous atmosphere scene element [:factorykey:`heterogeneous`].

    This class builds a one-dimensional heterogeneous atmosphere. It expands as
    a ``heterogeneous`` kernel plugin, which takes as parameters a set of
    paths to volume data files. The radiative properties used to configure
    :class:`.HeterogeneousAtmosphere` can be specified two ways:

    - if the ``profile`` field is specified, kernel volume data files will be
      created using those data;
    - if the ``profile`` field is not specified (*i.e.* set to ``None``), kernel
      volume data files will be read from locations set in the ``albedo_fname``
      and ``sigma_t_fname`` attributes (which then must be set to paths
      pointing to existing files).

    If ``profile`` is specified:

    - if ``albedo_fname`` and ``sigma_t_fname`` are specified, data files will
      be written to those paths;
    - if ``albedo_fname`` and ``sigma_t_fname`` are not specified (*i.e.* set
      to ``None``), filenames will be generated based on ``cache_dir``.

    .. note::

       Generated files are not destroyed after execution and can be accessed
       using paths saved in the ``albedo_fname`` and ``sigma_t_fname`` fields.

    .. warning::

       While radiative properties specified using the ``profile`` field will be
       scaled according to kernel default unit override, existing volume data
       will not.
    """

    profile = documented(
        attr.ib(
            default=None,
            converter=RadProfileFactory.convert,
            validator=attr.validators.optional(attr.validators.instance_of(RadProfile))
        ),
        doc="Radiative property profile used. If set, volume data files will be "
            "created from profile data to initialise the corresponding kernel "
            "plugin. If ``None``, :class:`.HeterogeneousAtmosphere` will assume "
            "that volume data files already exist.",
        type=":class:`~eradiate.radprops.rad_profile.RadProfile` or None",
        default="None",
    )

    albedo_fname = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(Path)
        ),
        doc="Path to the single scattering albedo volume data file. If "
            "``None``, a value will be created when the file will be "
            "requested.",
        type="path-like or None",
        default="None",
    )

    @albedo_fname.validator
    def _albedo_fname_validator(self, attribute, value):
        # The file should exist if no albedo value is provided to create it
        if self.profile is None:
            if value is None:
                raise ValueError("if 'profile' is not set, "
                                 "'albedo_fname' must be set")
            try:
                return is_file(self, attribute, value)
            except FileNotFoundError:
                raise

    sigma_t_fname = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(Path),
        ),
        doc="Path to the extinction coefficient volume data file. If ``None``, "
            "a value will be created when the file will be requested.",
        type="path-like or None",
        default="None",
    )

    @sigma_t_fname.validator
    def _sigma_t_fname_validator(self, attribute, value):
        # The file should exist if no sigma_t value is provided to create it
        if self.profile is None:
            if value is None:
                raise ValueError("if 'profile' is not set, "
                                 "'sigma_t_fname' must be set")
            try:
                return is_file(self, attribute, value)
            except FileNotFoundError:
                raise

    cache_dir = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(Path)
        ),
        doc="Path to a cache directory where volume data files will be "
            "created. If ``None``, a temporary cache directory will be used.",
        type="path-like or None",
        default="None",
    )

    _quantities = {
        "albedo": "albedo",
        "sigma_t": "collision_coefficient"
    }

    def __attrs_post_init__(self):
        # Check for automatic width and height computation
        if self.profile is not None:
            if self.toa_altitude != "auto":
                raise ValueError("toa_altitude must be set to 'auto' when "
                                 "profile is set")
        else:
            if self.toa_altitude == "auto":
                raise ValueError("toa_altitude cannot be set to 'auto' when "
                                 "profile is None")
            if self.width == "auto":
                raise ValueError("width cannot be set to 'auto' when profile "
                                 "is None")

        # Prepare cache directory in case we'd need it
        if self.cache_dir is None:
            self.cache_dir = Path(tempfile.mkdtemp())
        else:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def height(self):
        if self.toa_altitude == "auto":
            ds = self.profile.to_dataset()
            return ureg.Quantity(ds.z_level.values.max(), ds.z_level.units)
        else:
            return ureg.Quantity(100., ureg.km)

    @property
    def kernel_width(self):
        """Return scene width based on configuration."""

        if self.width == "auto":
            if self.profile is None:
                albedo = ureg.Quantity(
                    read_binary_grid3d(self.albedo_fname),
                    ureg.dimensionless
                )
                sigma_t = ureg.Quantity(
                    read_binary_grid3d(self.sigma_t_fname),
                    uck.get("collision_coefficient")
                )
            else:
                albedo = self.profile.albedo
                sigma_t = self.profile.sigma_t

            sigma_s = sigma_t * albedo
            min_sigma_s = sigma_s.min()
            if min_sigma_s > 0.:
                width = 10. / min_sigma_s
                if width > ureg.Quantity(1e3, "km"):
                    width = ureg.Quantity(1e3, "km")
            else:
                raise ValueError("cannot compute width automatically when "
                                 "scattering coefficient reaches zero")
        else:
            width = self.width

        return width

    def make_volume_data(self, fields=None):
        """Create volume data files for requested fields.

        Parameter ``fields`` (str or list[str] or None):
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
                field_fname = self.cache_dir / f"{field}.vol"
                setattr(self, f"{field}_fname", field_fname)

            # We have the data and the filename: we can create the file
            write_binary_grid3d(
                field_fname,
                field_quantity.to(uck.get(self._quantities[field])).magnitude
            )

    def phase(self):
        return {f"phase_{self.id}": {"type": "rayleigh"}}

    def media(self, ref=False):
        from eradiate.kernel.core import ScalarTransform4f

        k_width = self.kernel_width.to(uck.get("length")).magnitude
        k_height = self.kernel_height.to(uck.get("length")).magnitude
        k_offset = self.kernel_offset.to(uck.get("length")).magnitude

        # First, transform the [0, 1]^3 cube to the right dimensions
        trafo = ScalarTransform4f([
            [k_width, 0., 0., -0.5 * k_width],
            [0., k_width, 0., -0.5 * k_width],
            [0., 0., k_height + k_offset, -k_offset],
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

        k_length = uck.get("length")
        k_width = self.kernel_width.to(k_length).magnitude
        k_height = self.kernel_height.to(k_length).magnitude
        k_offset = self.kernel_offset.to(k_length).magnitude

        return {
            f"shape_{self.id}": {
                "type":
                    "cube",
                "to_world":
                    ScalarTransform4f([
                        [0.5 * k_width, 0., 0., 0.],
                        [0., 0.5 * k_width, 0., 0.],
                        [0., 0., 0.5 * k_height, 0.5 * k_height - k_offset],
                        [0., 0., 0., 1.],
                    ]),
                "bsdf": {
                    "type": "null"
                },
                "interior":
                    medium
            }
        }
