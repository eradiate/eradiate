from __future__ import annotations

import pathlib
import struct
from abc import ABC, abstractmethod
from typing import Dict, MutableMapping, Optional, Union

import attr
import numpy as np
import pint
import pinttr
import xarray as xr

from ..core import SceneElement
from ... import converters, validators
from ..._factory import Factory
from ...attrs import AUTO, documented, get_doc, parse_docs
from ...contexts import KernelDictContext
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

    id: Optional[str] = documented(
        attr.ib(
            default="atmosphere",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
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
        "Unit-enabled field (default unit: cdu[length]).",
        type="float or AUTO",
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

        Parameter ``ctx`` (:class:`.KernelDictContext`):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → :class:`~pint.Quantity`:
            Atmosphere's width.
        """
        pass

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @abstractmethod
    def kernel_phase(self, ctx: KernelDictContext) -> MutableMapping:
        """
        Return phase function plugin specifications only.

        Parameter ``ctx`` (:class:`.KernelDictContext`):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → dict:
            Return a dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the phase
            functions attached to the atmosphere.
        """
        pass

    @abstractmethod
    def kernel_media(self, ctx: KernelDictContext) -> MutableMapping:
        """
        Return medium plugin specifications only.

        Parameter ``ctx`` (:class:`.KernelDictContext`):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → dict:
            Return a dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the media
            attached to the atmosphere.
        """
        pass

    @abstractmethod
    def kernel_shapes(self, ctx: KernelDictContext) -> MutableMapping:
        """
        Return shape plugin specifications only.

        Parameter ``ctx`` (:class:`.KernelDictContext`):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → dict:
            A dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the shapes
            attached to the atmosphere.
        """
        pass

    def kernel_height(self, ctx: KernelDictContext) -> pint.Quantity:
        """
        Return the height of the kernel object delimiting the atmosphere.

        Parameter ``ctx`` (:class:`.KernelDictContext`):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → :class:`~pint.Quantity`:
            Height of the kernel object delimiting the atmosphere
        """
        return self.height + self.kernel_offset(ctx=ctx)

    def kernel_offset(self, ctx: KernelDictContext) -> pint.Quantity:
        """
        Return vertical offset used to position the kernel object delimiting the
        atmosphere. The created cuboid shape will be shifted towards negative
        Z values by this amount.

        .. note::

           This is required to ensure that the surface is the only shape
           which can be intersected at ground level during ray tracing.

        Parameter ``ctx`` (:class:`.KernelDictContext`):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → :class:`~pint.Quantity`:
            Vertical offset of cuboid shape.
        """
        return self.height * 1e-3

    def kernel_width(self, ctx: KernelDictContext) -> pint.Quantity:
        """
        Return width of the kernel object delimiting the atmosphere.

        Parameter ``ctx`` (:class:`.KernelDictContext`):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → :class:`~pint.Quantity`:
            Width of the kernel object delimiting the atmosphere.
        """
        return self.eval_width(ctx=ctx)

    def kernel_dict(self, ctx: KernelDictContext) -> Dict:
        kernel_dict = {}

        if not ctx.ref:
            kernel_dict[self.id] = self.kernel_shapes(ctx=ctx)[f"shape_{self.id}"]
        else:
            kernel_dict[f"phase_{self.id}"] = self.kernel_phase(ctx=ctx)[
                f"phase_{self.id}"
            ]
            kernel_dict[f"medium_{self.id}"] = self.kernel_media(ctx=ctx)[
                f"medium_{self.id}"
            ]
            kernel_dict[self.id] = self.kernel_shapes(ctx=ctx)[f"shape_{self.id}"]

        return kernel_dict


def write_binary_grid3d(
    filename: Union[str, pathlib.Path], values: Union[np.ndarray, xr.DataArray]
) -> None:
    """
    Write volume data to a binary file so that a ``gridvolume`` kernel plugin
    can be instantiated with that file.

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


def read_binary_grid3d(filename: str) -> np.ndarray:
    """
    Reads a volume data binary file.
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
