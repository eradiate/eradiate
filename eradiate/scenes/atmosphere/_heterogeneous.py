import struct
import tempfile
from pathlib import Path
from typing import List, MutableMapping, Optional, Union

import attr
import numpy as np
import pint
import xarray as xr

from eradiate.contexts import KernelDictContext, SpectralContext

from ._core import Atmosphere, AtmosphereFactory
from ...attrs import AUTO, documented, parse_docs
from ...kernel.transform import map_cube, map_unit_cube
from ...radprops import RadProfileFactory
from ...radprops.rad_profile import RadProfile, US76ApproxRadProfile
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


def write_binary_grid3d(filename, values):
    """
    Write volume data to a binary file so that a ``gridvolume`` kernel plugin can be
    instantiated with that file.

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
        raise TypeError(
            f"unsupported data type {type(values)} "
            f"(expected numpy array or xarray DataArray)"
        )

    if values.ndim not in {3, 4}:
        raise ValueError(
            f"'values' must have 3 or 4 dimensions " f"(got shape {values.shape})"
        )

    # note: this is an exact copy of the function write_binary_grid3d from
    # https://github.com/mitsuba-renderer/mitsuba-data/blob/master/tests/scenes/participating_media/create_volume_data.py

    with open(filename, "wb") as f:
        f.write(b"V")
        f.write(b"O")
        f.write(b"L")
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

    with open(filename, "rb") as f:
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


@AtmosphereFactory.register("heterogeneous")
@parse_docs
@attr.s
class HeterogeneousAtmosphere(Atmosphere):
    """
    Heterogeneous atmosphere scene element [:factorykey:`heterogeneous`].

    This class builds a one-dimensional heterogeneous atmosphere.
    It expands as a ``heterogeneous`` kernel plugin, which takes as parameters
    a phase function and a set of paths to volume data files.
    The radiative properties used to configure
    :class:`.HeterogeneousAtmosphere` are specified by a :class:`.RadProfile`
    object.
    The vertical extension of the atmosphere is automatically adjusted to
    match that of the :class:`.RadProfile` object.
    The atmosphere's bottom altitude is set to 0 km.
    The phase function is set to :class:`.RayleighPhaseFunction`.
    """

    profile: RadProfile = documented(
        attr.ib(
            default=attr.Factory(US76ApproxRadProfile),
            converter=RadProfileFactory.convert,
            validator=attr.validators.instance_of(RadProfile),
        ),
        doc="Radiative property profile used. If set, volume data files will be "
        "created from profile data to initialise the corresponding kernel "
        "plugin.",
        type=":class:`~eradiate.radprops.rad_profile.RadProfile`",
        default=":class:`US76ApproxRadProfile() <.US76ApproxRadProfile>`",
    )

    albedo_filename: str = documented(
        attr.ib(
            default="albedo.vol",
            converter=str,
            validator=attr.validators.instance_of(str),
        ),
        doc="Name of the albedo volume data file.",
        type="str",
        default='"albedo.vol"',
    )

    sigma_t_filename: str = documented(
        attr.ib(
            default="sigma_t.vol",
            converter=str,
            validator=attr.validators.instance_of(str),
        ),
        doc="Name of the extinction coefficient volume data file.",
        type="str",
        default='"sigma_t.vol"',
    )

    cache_dir: Path = documented(
        attr.ib(
            default=Path(tempfile.mkdtemp()),
            converter=Path,
            validator=attr.validators.instance_of(Path),
        ),
        doc="Path to a cache directory where volume data files will be created.",
        type="path-like",
        default="Temporary directory",
    )

    _quantities = {"albedo": "albedo", "sigma_t": "collision_coefficient"}

    def __attrs_post_init__(self):
        # Prepare cache directory in case we'd need it
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    #                             Properties
    # --------------------------------------------------------------------------

    @property
    def albedo_file(self) -> Path:
        return self.cache_dir / self.albedo_filename

    @property
    def sigma_t_file(self) -> Path:
        return self.cache_dir / self.sigma_t_filename

    @property
    def bottom(self) -> pint.Quantity:
        return ureg.Quantity(0.0, "km")

    @property
    def top(self) -> pint.Quantity:
        return self.profile.levels.max()

    # --------------------------------------------------------------------------
    #                       Evaluation methods
    # --------------------------------------------------------------------------

    def eval_width(self, ctx: KernelDictContext) -> pint.Quantity:
        if self.width is AUTO:
            spectral_ctx = ctx.spectral_ctx if ctx is not None else None

            if self.profile is None:
                albedo = ureg.Quantity(
                    read_binary_grid3d(self.albedo_filename),
                    ureg.dimensionless,
                )
                sigma_t = ureg.Quantity(
                    read_binary_grid3d(self.sigma_t_filename),
                    uck.get("collision_coefficient"),
                )
                min_sigma_s = (sigma_t * albedo).min()
            else:
                min_sigma_s = self.profile.sigma_s(spectral_ctx).min()

            if min_sigma_s <= 0.0:
                raise ValueError(
                    "cannot compute width automatically when scattering "
                    "coefficient reaches zero"
                )

            return min(10.0 / min_sigma_s, ureg.Quantity(1e3, "km"))

        else:
            return self.width

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    def make_volume_data(
        self,
        fields: Optional[Union[str, List[str]]] = None,
        spectral_ctx: Optional[SpectralContext] = None,
    ) -> None:
        """
        Create volume data files for requested fields.

        Parameter ``fields`` (str or list[str] or None):
            If str, field for which to create volume data file. If list,
            fields for which to create volume data files. If ``None``,
            all supported fields are processed (``["albedo", "sigma_t"]``).
            Default: ``None``.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext`):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).
        """
        if spectral_ctx is None:
            raise ValueError("keyword argument 'spectral_ctx' must be specified")

        supported_fields = set(self._quantities.keys())

        if fields is None:
            fields = supported_fields
        elif isinstance(fields, str):
            fields = {fields}

        if self.profile is None:
            raise ValueError("'profile' is not set, cannot write volume data " "files")

        for field in fields:
            # Is the requested field supported?
            if field not in supported_fields:
                raise ValueError(f"field {field} cannot be used to create volume data")

            # Does the considered field have values?
            field_quantity = getattr(self.profile, field)(spectral_ctx)

            if field_quantity is None:
                raise ValueError(f"field {field} is empty, cannot create volume data")

            # If file name is not specified, we create one
            field_fname = getattr(self, f"{field}_file")

            # We have the data and the filename: we can create the file
            field_quantity.m_as(uck.get(self._quantities[field]))
            write_binary_grid3d(
                field_fname,
                field_quantity.m_as(uck.get(self._quantities[field])),
            )

    def kernel_phase(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
        return {f"phase_{self.id}": {"type": "rayleigh"}}

    def kernel_media(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:

        length_units = uck.get("length")
        width = self.kernel_width(ctx).m_as(length_units)
        top = self.top.m_as(length_units)
        bottom = self.bottom.m_as(length_units)
        trafo = map_unit_cube(
            xmin=-width / 2.0,
            xmax=width / 2.0,
            ymin=-width / 2.0,
            ymax=width / 2.0,
            zmin=bottom,
            zmax=top,
        )

        self.make_volume_data("albedo", spectral_ctx=ctx.spectral_ctx)
        self.make_volume_data("sigma_t", spectral_ctx=ctx.spectral_ctx)

        return {
            f"medium_{self.id}": {
                "type": "heterogeneous",
                "phase": {"type": "rayleigh"},
                "sigma_t": {
                    "type": "gridvolume",
                    "filename": str(self.sigma_t_file),
                    "to_world": trafo,
                },
                "albedo": {
                    "type": "gridvolume",
                    "filename": str(self.albedo_file),
                    "to_world": trafo,
                },
            }
        }

    def kernel_shapes(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
        if ctx.ref:
            medium = {"type": "ref", "id": f"medium_{self.id}"}
        else:
            medium = self.kernel_media(ctx=None)[f"medium_{self.id}"]

        length_units = uck.get("length")
        width = self.kernel_width(ctx).m_as(length_units)
        bottom = self.bottom.m_as(length_units)
        top = self.top.m_as(length_units)
        offset = self.kernel_offset(ctx).m_as(length_units)
        trafo = map_cube(
            xmin=-width / 2.0,
            xmax=width / 2.0,
            ymin=-width / 2.0,
            ymax=width / 2.0,
            zmin=bottom - offset,
            zmax=top,
        )

        return {
            f"shape_{self.id}": {
                "type": "cube",
                "to_world": trafo,
                "bsdf": {"type": "null"},
                "interior": medium,
            }
        }
