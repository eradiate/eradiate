from __future__ import annotations

import pathlib
import struct
import tempfile
import typing as t
from abc import ABC, abstractmethod

import attr
import numpy as np
import pint
import pinttr
import xarray as xr

from ..core import KernelDict, SceneElement
from ... import converters
from ... import unit_context_kernel as uck
from ... import validators
from ..._factory import Factory
from ..._util import onedict_value
from ...attrs import AUTO, documented, get_doc, parse_docs
from ...contexts import KernelDictContext, SpectralContext
from ...kernel.transform import map_cube, map_unit_cube
from ...units import to_quantity
from ...units import unit_context_config as ucc

atmosphere_factory = Factory()


@parse_docs
@attr.s
class Atmosphere(SceneElement, ABC):
    """
    An abstract base class defining common facilities for all atmospheres.

    An atmosphere consists of a kernel medium (with a phase function) attached
    to a kernel shape.

    .. note::
       The shape type is restricted to cuboid shapes at the moment.
    """

    id: t.Optional[str] = documented(
        attr.ib(
            default="atmosphere",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        init_type=get_doc(SceneElement, "id", "init_type"),
        default='"atmosphere"',
    )

    width: pint.Quantity = documented(
        pinttr.ib(
            default=AUTO,
            converter=converters.auto_or(
                pinttr.converters.to_units(ucc.deferred("length"))
            ),
            validator=validators.auto_or(
                pinttr.validators.has_compatible_units, validators.is_positive
            ),
            units=ucc.deferred("length"),
        ),
        doc="Atmosphere width. If set to ``AUTO``, a value will be estimated to "
        "ensure that the medium is optically thick. The implementation of "
        "this estimate depends on the concrete class inheriting from this "
        "one.\n"
        "\n"
        "Unit-enabled field (default units: ucc['length']).",
        type="quantity or AUTO",
        init_type="quantity or float or AUTO",
        default="AUTO",
    )

    # --------------------------------------------------------------------------
    #                               Properties
    # --------------------------------------------------------------------------

    @property
    @abstractmethod
    def bottom(self) -> pint.Quantity:
        """
        Return the atmosphere's bottom altitude.

        Returns → :class:`~pint.Quantity`:
            Atmosphere's bottom altitude.
        """
        pass

    @property
    @abstractmethod
    def top(self) -> pint.Quantity:
        """
        Return the atmosphere's top altitude.

        Returns → :class:`~pint.Quantity`:
            Atmosphere's top altitude.
        """
        pass

    @property
    def height(self) -> pint.Quantity:
        """
        Return the atmosphere's height.

        Returns → :class:`~pint.Quantity`:
            Atmosphere's height.
        """
        return self.top - self.bottom

    # --------------------------------------------------------------------------
    #                           Evaluation methods
    # --------------------------------------------------------------------------

    @abstractmethod
    def eval_width(self, ctx: KernelDictContext) -> pint.Quantity:
        """
        Return the Atmosphere's width.

        Parameters
        ----------
        ctx : :class:`.KernelDictContext`
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns
        -------
        quantity
            Atmosphere width.
        """
        pass

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @abstractmethod
    def kernel_phase(self, ctx: KernelDictContext) -> KernelDict:
        """
        Return phase function plugin specifications only.

        Parameters
        ----------
        ctx : :class:`.KernelDictContext`
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns
        -------
        :class:`.KernelDict`
            A kernel dictionary containing all the phase functions attached to
            the atmosphere.
        """
        pass

    @abstractmethod
    def kernel_media(self, ctx: KernelDictContext) -> KernelDict:
        """
        Return medium plugin specifications only.

        Parameters
        ----------
        ctx : :class:`.KernelDictContext`
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns
        -------
        :class:`.KernelDict`
            A kernel dictionary containing all the media attached to the
            atmosphere.
        """
        pass

    @abstractmethod
    def kernel_shapes(self, ctx: KernelDictContext) -> KernelDict:
        """
        Return shape plugin specifications only.

        Parameters
        ----------
        ctx : :class:`.KernelDictContext`
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns
        -------
        :class:`.KernelDict`
            A kernel dictionary containing all the shapes attached to the
            atmosphere.
        """
        pass

    def kernel_height(self, ctx: KernelDictContext) -> pint.Quantity:
        """
        Return the height of the kernel object delimiting the atmosphere.

        Parameters
        ----------
        ctx : :class:`.KernelDictContext`
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns
        -------
        quantity
            Height of the kernel object delimiting the atmosphere
        """
        return self.height + self.kernel_offset(ctx=ctx)

    def kernel_offset(self, ctx: KernelDictContext) -> pint.Quantity:
        """
        Return vertical offset used to position the kernel object delimiting the
        atmosphere. The created cuboid shape will be shifted towards negative
        Z values by this amount.

        Parameters
        ----------
        ctx : :class:`.KernelDictContext`
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns
        -------
        quantity
            Vertical offset of cuboid shape.

        Notes
        -----
        This offset is required to ensure that the surface is the only shape
        which can be intersected at ground level during ray tracing.
        """
        return self.height * 1e-3

    def kernel_width(self, ctx: KernelDictContext) -> pint.Quantity:
        """
        Return width of the kernel object delimiting the atmosphere.

        Parameters
        ----------
        ctx : :class:`.KernelDictContext`
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns
        -------
        quantity
            Width of the kernel object delimiting the atmosphere.
        """
        return self.eval_width(ctx=ctx)

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        kernel_dict = KernelDict()

        if ctx.ref:
            kernel_phase = self.kernel_phase(ctx=ctx)
            kernel_dict.data[f"phase_{self.id}"] = kernel_phase[f"phase_{self.id}"]
            kernel_media = self.kernel_media(ctx=ctx)
            kernel_dict.data[f"medium_{self.id}"] = kernel_media[f"medium_{self.id}"]

        kernel_shapes = self.kernel_shapes(ctx=ctx)
        kernel_dict.data[self.id] = kernel_shapes[f"shape_{self.id}"]

        return kernel_dict


def write_binary_grid3d(
    filename: t.Union[str, pathlib.Path], values: t.Union[np.ndarray, xr.DataArray]
) -> None:
    """
    Write volume data to a binary file so that a ``gridvolume`` kernel plugin
    can be instantiated with that file.

    Parameters
    ----------
    filename : path-like
        File name.

    Parameters
    ----------
    values : ndarray or DataArray
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


def read_binary_grid3d(filename: str) -> np.ndarray:
    """
    Reads a volume data binary file.

    Parameters
    ----------
    filename : str
        File name.

    Returns
    -------
    ndarray
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


@parse_docs
@attr.s
class AbstractHeterogeneousAtmosphere(Atmosphere, ABC):
    """
    Heterogeneous atmosphere abstract base class. This class defines the basic
    interface common to all heterogeneous atmosphere models.
    """

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

    cache_dir: pathlib.Path = documented(
        attr.ib(
            default=pathlib.Path(tempfile.mkdtemp()),
            converter=pathlib.Path,
            validator=attr.validators.instance_of(pathlib.Path),
        ),
        doc="Path to a cache directory where volume data files will be created.",
        type="path-like",
        default="Temporary directory",
    )

    def __attrs_post_init__(self) -> None:
        self.update()

    def update(self) -> None:
        """
        Update internal state.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    #                        Volume data files
    # --------------------------------------------------------------------------

    @property
    def albedo_file(self) -> pathlib.Path:
        return self.cache_dir / self.albedo_filename

    @property
    def sigma_t_file(self) -> pathlib.Path:
        return self.cache_dir / self.sigma_t_filename

    # --------------------------------------------------------------------------
    #                    Spatial and thermophysical properties
    # --------------------------------------------------------------------------

    # Nothing at the moment

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    @abstractmethod
    def eval_radprops(self, spectral_ctx: SpectralContext) -> xr.Dataset:
        """
        Return a dataset that holds the radiative properties profile of this
        atmospheric model.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Returns
        -------
        Dataset
            Radiative properties profile dataset.
        """
        pass

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    def kernel_media(self, ctx: KernelDictContext) -> KernelDict:
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

        radprops = self.eval_radprops(spectral_ctx=ctx.spectral_ctx)
        albedo = to_quantity(radprops.albedo).m_as(uck.get("albedo"))
        sigma_t = to_quantity(radprops.sigma_t).m_as(uck.get("collision_coefficient"))

        write_binary_grid3d(
            filename=str(self.albedo_file), values=albedo[np.newaxis, np.newaxis, ...]
        )

        write_binary_grid3d(
            filename=str(self.sigma_t_file), values=sigma_t[np.newaxis, np.newaxis, ...]
        )

        if ctx.ref:
            phase = {"type": "ref", "id": f"phase_{self.id}"}
        else:
            phase = onedict_value(self.kernel_phase(ctx=ctx))

        return KernelDict(
            {
                f"medium_{self.id}": {
                    "type": "heterogeneous",
                    "phase": phase,
                    "albedo": {
                        "type": "gridvolume",
                        "filename": str(self.albedo_file),
                        "to_world": trafo,
                    },
                    "sigma_t": {
                        "type": "gridvolume",
                        "filename": str(self.sigma_t_file),
                        "to_world": trafo,
                    },
                }
            }
        )

    def kernel_shapes(self, ctx: KernelDictContext) -> KernelDict:
        if ctx.ref:
            medium = {"type": "ref", "id": f"medium_{self.id}"}
        else:
            medium = self.kernel_media(ctx)[f"medium_{self.id}"]

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

        return KernelDict(
            {
                f"shape_{self.id}": {
                    "type": "cube",
                    "to_world": trafo,
                    "bsdf": {"type": "null"},
                    "interior": medium,
                }
            }
        )
