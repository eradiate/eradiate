from __future__ import annotations

import pathlib
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
from ...kernel.gridvolume import write_binary_grid3d
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

    Notes
    -----
    The only allowed stencil for :class:`.Atmosphere` objects is currently a
    cuboid.
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
        pint.Quantity: Atmosphere bottom altitude.
        """
        pass

    @property
    @abstractmethod
    def top(self) -> pint.Quantity:
        """
        pint.Quantity: Atmosphere top altitude.
        """
        pass

    @property
    def height(self) -> pint.Quantity:
        """
        pint.Quantity: Atmosphere height.
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
            kernel_dict.data[f"phase_{self.id}"] = onedict_value(kernel_phase)
            kernel_media = self.kernel_media(ctx=ctx)
            kernel_dict.data[f"medium_{self.id}"] = onedict_value(kernel_media)

        kernel_shapes = self.kernel_shapes(ctx=ctx)
        kernel_dict.data[self.id] = onedict_value(kernel_shapes)

        return kernel_dict


@parse_docs
@attr.s
class AbstractHeterogeneousAtmosphere(Atmosphere, ABC):
    """
    Heterogeneous atmosphere abstract base class. This class defines the basic
    interface common to all heterogeneous atmosphere models.
    """

    _sigma_t_filename: t.Optional[str] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(str),
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc="Name of the extinction coefficient volume data file. If unset, a "
        "file name will be generated automatically.",
        type="str or None",
        init_type="str, optional",
    )

    _albedo_filename: str = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(str),
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc="Name of the albedo volume data file. If unset, a file name will "
        "be generated automatically.",
        type="str or None",
        init_type="str, optional",
    )

    cache_dir: pathlib.Path = documented(
        attr.ib(
            factory=lambda: pathlib.Path(tempfile.mkdtemp()),
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
    def sigma_t_filename(self) -> str:
        """
        str: Name of the extinction coefficient volume data file.
        """
        return (
            self._sigma_t_filename
            if self._sigma_t_filename is not None
            else f"{self.id}_sigma_t.vol"
        )

    @property
    def sigma_t_file(self) -> pathlib.Path:
        """
        path: Absolute path to the extinction coefficient volume data file
        """
        return self.cache_dir / self.sigma_t_filename

    @property
    def albedo_filename(self) -> str:
        """
        str: Name of the albedo volume data file.
        """
        return (
            self._albedo_filename
            if self._albedo_filename is not None
            else f"{self.id}_sigma_t.vol"
        )

    @property
    def albedo_file(self) -> pathlib.Path:
        """
        path: Absolute path to the albedo volume data file.
        """
        return self.cache_dir / self.albedo_filename

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

    def _shape_transform(
        self, ctx: KernelDictContext
    ) -> "mitsuba.core.ScalarTransform4f":
        length_units = uck.get("length")
        width = self.kernel_width(ctx).m_as(length_units)
        bottom = self.bottom.m_as(length_units)
        top = self.top.m_as(length_units)
        offset = self.kernel_offset(ctx).m_as(length_units)

        return map_cube(
            xmin=-0.5 * width,
            xmax=0.5 * width,
            ymin=-0.5 * width,
            ymax=0.5 * width,
            zmin=bottom - offset,
            zmax=top,
        )

    def _gridvolume_transform(
        self, ctx: KernelDictContext
    ) -> "mitsuba.core.ScalarTransform4f":
        length_units = uck.get("length")
        width = self.kernel_width(ctx).m_as(length_units)
        top = self.top.m_as(length_units)
        bottom = self.bottom.m_as(length_units)

        return map_unit_cube(
            xmin=-0.5 * width,
            xmax=0.5 * width,
            ymin=-0.5 * width,
            ymax=0.5 * width,
            zmin=bottom,
            zmax=top,
        )

    def kernel_media(self, ctx: KernelDictContext) -> KernelDict:
        radprops = self.eval_radprops(spectral_ctx=ctx.spectral_ctx)
        albedo = radprops.albedo.values
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

        trafo = self._gridvolume_transform(ctx)

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

        return KernelDict(
            {
                f"shape_{self.id}": {
                    "type": "cube",
                    "to_world": self._shape_transform(ctx),
                    "bsdf": {"type": "null"},
                    "interior": medium,
                }
            }
        )
