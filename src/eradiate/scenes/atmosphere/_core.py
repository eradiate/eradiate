from __future__ import annotations

import pathlib
import tempfile
import typing as t
from abc import ABC, abstractmethod

import attr
import mitsuba as mi
import numpy as np
import pint
import pinttr
import xarray as xr

from ..core import KernelDict, SceneElement
from ..shapes import CuboidShape, SphereShape
from ... import converters
from ... import unit_context_kernel as uck
from ... import validators
from ..._factory import Factory
from ..._util import onedict_value
from ...attrs import AUTO, AutoType, documented, get_doc, parse_docs
from ...contexts import KernelDictContext, SpectralContext
from ...kernel.gridvolume import write_binary_grid3d
from ...kernel.transform import map_unit_cube
from ...units import to_quantity
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg

atmosphere_factory = Factory()


@parse_docs
@attr.s
class AtmosphereGeometry:
    """
    Base class defining the geometry of the atmosphere.
    """

    @classmethod
    def convert(cls, value: t.Any) -> t.Any:
        """
        Attempt conversion of a value to a :class:`AtmosphereGeometry` subtype.

        Parameters
        ----------
        value
            Value to attempt conversion of. If a dictionary is passed, its
            ``"type"`` key is used to route its other entries as keyword
            arguments to the appropriate subtype's constructor. If a string is
            passed, this method calls itself with the parameter
            ``{"type": value}``.

        Returns
        -------
        result
            If `value` is a dictionary, the constructed
            :class:`.AtmosphereGeometry` instance is returned. Otherwise,
            `value` is returned.

        Raises
        ------
        ValueError
            A dictionary was passed but the requested type is unknown.
        """
        if isinstance(value, str):
            return cls.convert({"type": value})

        if isinstance(value, dict):
            value = value.copy()
            geometry_type = value.pop("type")

            # Note: if this conditional becomes large, use a dictionary
            if geometry_type == "plane_parallel":
                geometry_cls = PlaneParallelGeometry
            elif geometry_type == "spherical_shell":
                geometry_cls = SphericalShellGeometry
            else:
                raise ValueError(f"unknown geometry type '{geometry_type}'")

            return geometry_cls(**pinttr.interpret_units(value, ureg=ureg))

        return value


@parse_docs
@attr.s
class PlaneParallelGeometry(AtmosphereGeometry):
    """
    Plane parallel geometry.

    A plane parallel atmosphere is translation-invariant in the X and Y
    directions. However, Eradiate represents it with a finite 3D geometry
    consisting of a cuboid. By default, the cuboid's size is computed
    automatically; however, it can also be forced by assigning a value to
    the `width` field.
    """

    width: t.Union[pint.Quantity, AutoType] = documented(
        pinttr.ib(
            default=AUTO,
            converter=converters.auto_or(
                pinttr.converters.to_units(ucc.deferred("length"))
            ),
            validator=validators.auto_or(pinttr.validators.has_compatible_units),
            units=ucc.deferred("length"),
        ),
        doc="Cuboid shape width.",
        type="quantity or AUTO",
        init_type="quantity or float",
        default="AUTO",
    )


@attr.s
class SphericalShellGeometry(AtmosphereGeometry):
    """
    Spherical shell geometry.

    A spherical shell atmosphere has a spherical symmetry. Eradiate represents
    it with a finite 3D geometry consisting of a sphere. By default, the
    sphere's radius is set equal to Earth's radius.
    """

    planet_radius: pint.Quantity = documented(
        pinttr.ib(default=6378.1 * ureg.km, units=ucc.deferred("length")),
        doc="Planet radius. Defaults to Earth's radius.",
        type="quantity",
        init_type="quantity or float",
        default="6378.1 km",
    )


@parse_docs
@attr.s
class Atmosphere(SceneElement, ABC):
    """
    An abstract base class defining common facilities for all atmospheres.

    An atmosphere consists of a kernel medium (with a phase function) attached
    to a kernel shape.
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

    geometry: t.Optional[AtmosphereGeometry] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(AtmosphereGeometry.convert),
            validator=attr.validators.optional(
                attr.validators.instance_of(AtmosphereGeometry)
            ),
        ),
        doc="Parameters defining the basic geometry of the atmosphere.",
        type=".AtmosphereGeometry or None",
        init_type=".AtmosphereGeometry or dict or str, optional",
        default="None",
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
    #                    Spatial and thermophysical properties
    # --------------------------------------------------------------------------

    @abstractmethod
    def eval_mfp(self, ctx: KernelDictContext) -> pint.Quantity:
        """
        Compute a typical scattering mean free path. This rough estimate can be
        used *e.g.* to compute a distance guaranteeing that the medium can be
        considered optically thick.

        Parameters
        ----------
        ctx : :class:`.KernelDictContext`
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns
        -------
        mfp : quantity
            Mean free path estimate.
        """
        pass

    def eval_shape(self, ctx: KernelDictContext) -> t.Union[CuboidShape, SphereShape]:
        """
        Return the shape enclosing the atmosphere's volume.

        Parameters
        ----------
        ctx : .KernelDictContext
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns
        -------
        shape : .CuboidShape or .SphereShape
            Computed shape used as the medium stencil for kernel dictionary
            generation.
        """

        if isinstance(self.geometry, PlaneParallelGeometry):
            width = (
                self.geometry.width
                if self.geometry.width is not AUTO
                else self.kernel_width_plane_parallel(ctx)
            )

            return CuboidShape.atmosphere(top=self.top, bottom=self.bottom, width=width)

        elif isinstance(self.geometry, SphericalShellGeometry):
            planet_radius = self.geometry.planet_radius

            return SphereShape.atmosphere(top=self.top, planet_radius=planet_radius)

        elif self.geometry is None:
            raise ValueError(
                "The 'geometry' field must be an AtmosphereGeometry for "
                "kernel dictionary generation to work (got None)."
            )

        else:  # Shouldn't happen, prevented by validator
            raise TypeError(
                f"unhandled atmosphere geometry type '{type(self.geometry).__name__}'"
            )

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def id_shape(self):
        """
        str: Kernel dictionary key of the atmosphere's shape object.
        """
        return f"shape_{self.id}"

    @property
    def id_medium(self):
        """
        str: Kernel dictionary key of the atmosphere's medium object.
        """
        return f"medium_{self.id}"

    @property
    def id_phase(self):
        """
        str: Kernel dictionary key of the atmosphere's phase function object.
        """
        return f"phase_{self.id}"

    def kernel_width_plane_parallel(self, ctx: KernelDictContext) -> pint.Quantity:
        """
        When using a plane parallel geometry, compute the size of the cuboid
        shape enclosing the participating medium representing the atmosphere.

        Parameters
        ----------
        ctx : .KernelDictContext
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns
        -------
        width : quantity
            Computed cuboid width, eval to
            :meth:`10 / self.eval_mfp(ctx) <.eval_mfp>` if `
            `self.geometry.width`` is set to ``AUTO``, ``self.geometry.width``
            otherwise.

        Raises
        ------
        ValueError
            When a geometry other than :class:`.PlaneParallelGeometry` is used.

        """
        if not isinstance(self.geometry, PlaneParallelGeometry):
            raise ValueError(
                "Cuboid shape width is only relevant when using 'PlaneParallelGeometry' "
                f"(currently using a '{type(self.geometry).__name__}')"
            )

        if self.geometry.width is AUTO:
            mfp = self.eval_mfp(ctx)
            if mfp.magnitude == np.inf:
                return 1e7 * ureg.m  # default atmosphere width value
            else:
                return 10.0 * mfp
        else:
            return self.geometry.width

    @abstractmethod
    def kernel_phase(self, ctx: KernelDictContext) -> KernelDict:
        """
        Return phase function plugin specifications only.

        Parameters
        ----------
        ctx : .KernelDictContext
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns
        -------
        kernel_dict : .KernelDict
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
        ctx : .KernelDictContext
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns
        -------
        kernel_dict : .KernelDict
            A kernel dictionary containing all the media attached to the
            atmosphere.
        """
        pass

    def kernel_shapes(self, ctx: KernelDictContext) -> KernelDict:
        """
        Return shape plugin specifications only.

        Parameters
        ----------
        ctx : .KernelDictContext
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns
        -------
        kernel_dict : .KernelDict
            A kernel dictionary containing all the shapes attached to the
            atmosphere.
        """
        return self.eval_shape(ctx).kernel_dict(ctx)

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        # Inherit docstring

        # Note: Order matters! Mitsuba processes this dictionary in the order it
        # is iterated on.
        result = {
            self.id_phase: onedict_value(self.kernel_phase(ctx=ctx)),
            self.id_medium: onedict_value(self.kernel_media(ctx=ctx)),
            self.id_shape: onedict_value(self.kernel_shapes(ctx=ctx)),
        }

        result[self.id_medium].update({"phase": {"type": "ref", "id": self.id_phase}})
        result[self.id_shape].update(
            {
                "bsdf": {"type": "null"},
                "interior": {"type": "ref", "id": self.id_medium},
            }
        )

        return KernelDict(result)


@parse_docs
@attr.s
class AbstractHeterogeneousAtmosphere(Atmosphere, ABC):
    """
    Heterogeneous atmosphere base class. This class defines the basic interface
    common to all heterogeneous atmosphere models.
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
        default="None",
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
        default="None",
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

    scale: t.Optional[float] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(float),
            validator=attr.validators.optional(attr.validators.instance_of(float)),
        ),
        doc="If set, the extinction coefficient is scaled by the corresponding "
        "amount during computation.",
        type="float or None",
        init_type="float, optional",
        default="None",
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
            else f"{self.id}_albedo.vol"
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
        spectral_ctx : .SpectralContext
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
        # Inherit docstring

        # Collect extinction properties
        radprops = self.eval_radprops(spectral_ctx=ctx.spectral_ctx)
        albedo = radprops.albedo.values
        sigma_t = to_quantity(radprops.sigma_t).m_as(uck.get("collision_coefficient"))

        # Define volume data sources
        length_units = uck.get("length")
        top = self.top.m_as(length_units)
        bottom = self.bottom.m_as(length_units)

        if isinstance(self.geometry, PlaneParallelGeometry):
            write_binary_grid3d(
                filename=str(self.albedo_file),
                values=np.reshape(albedo, (-1, 1, 1)),
            )
            write_binary_grid3d(
                filename=str(self.sigma_t_file),
                values=np.reshape(sigma_t, (-1, 1, 1)),
            )

            width = self.kernel_width_plane_parallel(ctx).m_as(length_units)
            to_world = map_unit_cube(
                xmin=-0.5 * width,
                xmax=0.5 * width,
                ymin=-0.5 * width,
                ymax=0.5 * width,
                zmin=bottom,
                zmax=top,
            )

            volumes = {
                "albedo": {
                    "type": "gridvolume",
                    "filename": str(self.albedo_file),
                    "to_world": to_world,
                },
                "sigma_t": {
                    "type": "gridvolume",
                    "filename": str(self.sigma_t_file),
                    "to_world": to_world,
                },
            }

        elif isinstance(self.geometry, SphericalShellGeometry):
            write_binary_grid3d(
                filename=str(self.albedo_file),
                values=np.reshape(albedo, (1, 1, -1)),
            )
            write_binary_grid3d(
                filename=str(self.sigma_t_file),
                values=np.reshape(sigma_t, (1, 1, -1)),
            )

            planet_radius = self.geometry.planet_radius.m_as(length_units)
            rmax = planet_radius + top
            to_world = mi.ScalarTransform4f.scale(rmax)

            volumes = {
                "albedo": {
                    "type": "sphericalcoordsvolume",
                    "volume": {
                        "type": "gridvolume",
                        "filename": str(self.albedo_file),
                    },
                    "to_world": to_world,
                    "rmin": planet_radius / rmax,
                },
                "sigma_t": {
                    "type": "sphericalcoordsvolume",
                    "volume": {
                        "type": "gridvolume",
                        "filename": str(self.sigma_t_file),
                    },
                    "to_world": to_world,
                    "rmin": planet_radius / rmax,
                },
            }

        elif self.geometry is None:
            raise ValueError(
                "The 'geometry' field must be an AtmosphereGeometry for "
                "kernel dictionary generation to work (got None)."
            )

        else:  # Shouldn't happen, prevented by validator
            raise ValueError(
                f"unhandled atmosphere geometry type '{type(self.geometry).__name__}'"
            )

        # Create medium dictionary
        medium_dict = {
            "type": "heterogeneous",
            **volumes
            # Note: "phase" is deliberately unset, this is left to kernel_dict()
        }

        if self.scale is not None:
            medium_dict["scale"] = self.scale

        # Wrap it in a KernelDict
        return KernelDict({self.id_medium: medium_dict})
