from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

import attrs
import mitsuba as mi
import numpy as np
import pint
import pinttr
import xarray as xr

from ..core import CompositeSceneElement, Param, ParamFlags, SceneElement, traverse
from ..shapes import CuboidShape, SphereShape
from ..._factory import Factory
from ...attrs import documented, get_doc, parse_docs
from ...contexts import KernelDictContext, SpectralContext
from ...kernel.transform import map_unit_cube
from ...units import symbol
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg
from ...util.misc import flatten

atmosphere_factory = Factory()
atmosphere_factory.register_lazy_batch(
    [
        (
            "_heterogeneous.HeterogeneousAtmosphere",
            "heterogeneous",
            {},
        ),
        (
            "_homogeneous.HomogeneousAtmosphere",
            "homogeneous",
            {},
        ),
        (
            "_molecular_atmosphere.MolecularAtmosphere",
            "molecular",
            {"dict_constructor": "afgl_1986"},
        ),
        (
            "_particle_layer.ParticleLayer",
            "particle_layer",
            {},
        ),
    ],
    cls_prefix="eradiate.scenes.atmosphere",
)


@parse_docs
@attrs.define
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
@attrs.define
class PlaneParallelGeometry(AtmosphereGeometry):
    """
    Plane parallel geometry.

    A plane parallel atmosphere is translation-invariant in the X and Y
    directions. However, Eradiate represents it with a finite 3D geometry
    consisting of a cuboid. By default, the cuboid's size is computed
    automatically; however, it can also be forced by assigning a value to
    the `width` field.
    """

    width: pint.Quantity = documented(
        pinttr.field(default=1e7 * ureg.m, units=ucc.deferred("length")),
        doc="Cuboid shape width.",
        type="quantity",
        init_type="quantity or float",
        default="1e7 m",
    )


@parse_docs
@attrs.define
class SphericalShellGeometry(AtmosphereGeometry):
    """
    Spherical shell geometry.

    A spherical shell atmosphere has a spherical symmetry. Eradiate represents
    it with a finite 3D geometry consisting of a sphere. By default, the
    sphere's radius is set equal to Earth's radius.
    """

    planet_radius: pint.Quantity = documented(
        pinttr.field(default=6378.1 * ureg.km, units=ucc.deferred("length")),
        doc="Planet radius. Defaults to Earth's radius.",
        type="quantity",
        init_type="quantity or float",
        default="6378.1 km",
    )


@attrs.define(eq=False)
class ZGrid:
    """
    This class simply provides a hashable container for an altitude grid.
    This is required to allow for using the altitude as an argument of a
    LRU-cached function.
    """

    levels: pint.Quantity = pinttr.field(
        units=ucc.deferred("length"),
    )

    _layers: pint.Quantity = pinttr.field(
        init=False,
        default=None,
        units=ucc.deferred("length"),
    )

    _layer_height: pint.Quantity = pinttr.field(
        init=False,
        default=None,
        units=ucc.deferred("length"),
    )

    def __attrs_post_init__(self):
        self.update()

    def update(self):
        self._layer_height = np.diff(self.levels)
        self._layers = self.levels[:-1] + self._layer_height / 2

    @property
    def layers(self):
        return self._layers

    @property
    def layer_heights(self):
        return self._layer_height

    def n_levels(self):
        return len(self.levels)

    def n_layers(self):
        return len(self.layers)


@parse_docs
@attrs.define(eq=False, slots=False)
class Atmosphere(CompositeSceneElement, ABC):
    """
    An abstract base class defining common facilities for all atmospheres.

    An atmosphere consists of a kernel medium (with a phase function) attached
    to a kernel shape.
    """

    id: t.Optional[str] = documented(
        attrs.field(
            default="atmosphere",
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        init_type=get_doc(SceneElement, "id", "init_type"),
        default='"atmosphere"',
    )

    geometry: AtmosphereGeometry = documented(
        attrs.field(
            default="plane_parallel",
            converter=AtmosphereGeometry.convert,
            validator=attrs.validators.instance_of(AtmosphereGeometry),
        ),
        doc="Parameters defining the basic geometry of the atmosphere.",
        type=".AtmosphereGeometry",
        init_type=".AtmosphereGeometry or dict or str, optional",
        default='"plane_parallel"',
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

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def _shape(self) -> t.Union[CuboidShape, SphereShape]:
        """
        Return the shape associated with this atmosphere, factoring in the
        geometry and radiative property profile.
        """
        if isinstance(self.geometry, PlaneParallelGeometry):
            return CuboidShape.atmosphere(
                top=self.top, bottom=self.bottom, width=self.geometry.width
            )

        elif isinstance(self.geometry, SphericalShellGeometry):
            return SphereShape.atmosphere(
                top=self.top, planet_radius=self.geometry.planet_radius
            )

        else:  # Shouldn't happen, prevented by validator
            raise TypeError(
                f"unhandled atmosphere geometry type '{type(self.geometry).__name__}'"
            )

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

    @property
    @abstractmethod
    def _template_phase(self) -> dict:
        pass

    @property
    @abstractmethod
    def _template_medium(self) -> dict:
        pass

    @property
    def _template_shape(self) -> dict:
        """
        Return the shape enclosing the atmosphere's volume.

        Returns
        -------
        shape : .CuboidShape or .SphereShape
            Computed shape used as the medium stencil for kernel dictionary
            generation.
        """
        template, _ = traverse(self._shape)
        return template.data

    @property
    def template(self) -> dict:
        result = flatten(
            {
                self.id_phase: self._template_phase,
                self.id_medium: self._template_medium,
                self.id_shape: self._template_shape,
            }
        )

        result.update(
            {
                f"{self.id_medium}.phase.type": "ref",
                f"{self.id_medium}.phase.id": self.id_phase,
                f"{self.id_shape}.bsdf.type": "null",
                f"{self.id_shape}.interior.type": "ref",
                f"{self.id_shape}.interior.id": self.id_medium,
            }
        )

        return result

    @property
    def _params_phase(self) -> t.Dict[str, Param]:
        return {}

    @property
    def _params_medium(self) -> t.Dict[str, Param]:
        return {}

    @property
    def _params_shape(self) -> t.Dict[str, Param]:
        return {}

    @property
    def params(self) -> t.Dict[str, Param]:
        # Inherit docstring
        return flatten(
            {
                self.id_medium: {
                    **self._params_medium,
                    "phase_function": self._params_phase,
                },
                self.id_shape: self._params_shape,
            }
        )


@parse_docs
@attrs.define(eq=False, slots=False)
class AbstractHeterogeneousAtmosphere(Atmosphere, ABC):
    """
    Heterogeneous atmosphere base class. This class defines the basic interface
    common to all heterogeneous atmosphere models.
    """

    scale: t.Optional[float] = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(float),
            validator=attrs.validators.optional(attrs.validators.instance_of(float)),
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
        pass

    # --------------------------------------------------------------------------
    #                    Spatial and thermophysical properties
    # --------------------------------------------------------------------------

    @property
    @abstractmethod
    def zgrid(self) -> ZGrid:
        """
        ZGrid: Altitude grid at which thermophysical and radiative
               properties are evaluated. Corresponds to layer centres.
        """
        pass

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    def eval_radprops(
        self, sctx: SpectralContext, optional_fields: bool = False
    ) -> xr.Dataset:
        """
        Evaluate the extinction coefficients and albedo profiles.

        Parameters
        ----------
        sctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        optional_fields : bool, optional, default: False
            If ``True``, also output the absorption and scattering coefficients,
            not required for scene setup but useful for analysis and debugging.

        Returns
        -------
        Dataset
            A dataset containing with the following variables for the specified
            spectral context:

            * ``sigma_t``: extinction coefficient;
            * ``albedo``: albedo;
            * ``sigma_a``: absorption coefficient (optional);
            * ``sigma_s``: scattering coefficient (optional).

            Coordinates are the following:

            * ``z``: altitude.
        """
        sigma_units = ucc.get("collision_coefficient")
        sigma_t = self.eval_sigma_t(sctx).m_as(sigma_units)
        albedo = self.eval_albedo(sctx, None).magnitude

        data_vars = {
            "sigma_t": (
                "z_layer",
                sigma_t,
                {
                    "units": f"{symbol(sigma_units)}",
                    "standard_name": "extinction_coefficient",
                    "long_name": "extinction coefficient",
                },
            ),
            "albedo": (
                "z_layer",
                albedo,
                {
                    "standard_name": "albedo",
                    "long_name": "albedo",
                    "units": "",
                },
            ),
        }

        if optional_fields:
            data_vars.update(
                {
                    "sigma_a": (
                        "z_layer",
                        sigma_t * (1.0 - albedo),
                        {
                            "units": f"{symbol(sigma_units)}",
                            "standard_name": "absorption_coefficient",
                            "long_name": "absorption coefficient",
                        },
                    ),
                    "sigma_s": (
                        "z_layer",
                        sigma_t * albedo,
                        {
                            "units": f"{symbol(sigma_units)}",
                            "standard_name": "scattering_coefficient",
                            "long_name": "scattering coefficient",
                        },
                    ),
                }
            )

        return xr.Dataset(
            data_vars,
            coords={
                "z_layer": (
                    "z_layer",
                    self.zgrid.layers.magnitude,
                    {
                        "units": f"{symbol(self.zgrid.layers.units)}",
                        "standard_name": "layer_altitude",
                        "long_name": "layer altitude",
                    },
                )
            },
        )

    @abstractmethod
    def eval_albedo(
        self, sctx: SpectralContext, zgrid: t.Optional[ZGrid] = None
    ) -> pint.Quantity:
        """
        Evaluate albedo spectrum based on a spectral context. This method
        dispatches evaluation to specialised methods depending on the active
        mode.

        Parameters
        ----------
        sctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Returns
        -------
        quantity
            Evaluated spectrum as an array with length equal to the number of
            layers.
        """

        pass

    @abstractmethod
    def eval_sigma_t(
        self, sctx: SpectralContext, zgrid: t.Optional[ZGrid] = None
    ) -> pint.Quantity:
        """
        Evaluate extinction coefficient given a spectral context.

        Parameters
        ----------
        sctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Returns
        -------
        quantity
            Particle layer extinction coefficient.
        """

        pass

    @abstractmethod
    def eval_sigma_a(
        self, sctx: SpectralContext, zgrid: t.Optional[ZGrid] = None
    ) -> pint.Quantity:
        """
        Evaluate absorption coefficient given a spectral context.

        Parameters
        ----------
        sctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Returns
        -------
        quantity
            Particle layer extinction coefficient.
        """

        pass

    @abstractmethod
    def eval_sigma_s(
        self, sctx: SpectralContext, zgrid: t.Optional[ZGrid] = None
    ) -> pint.Quantity:
        """
        Evaluate scattering coefficient given a spectral context.

        Parameters
        ----------
        sctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Returns
        -------
        quantity
            Particle layer scattering coefficient.
        """

        pass

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def _template_medium(self) -> dict:
        length_units = uck.get("length")
        top = self.top.m_as(length_units)
        bottom = self.bottom.m_as(length_units)

        if isinstance(self.geometry, PlaneParallelGeometry):

            def eval_albedo(ctx: KernelDictContext):
                return mi.VolumeGrid(
                    np.reshape(
                        self.eval_albedo(ctx.spectral_ctx, None), (-1, 1, 1)
                    ).astype(np.float32)
                )

            def eval_sigma_t(ctx: KernelDictContext):
                return mi.VolumeGrid(
                    np.reshape(
                        self.eval_sigma_t(ctx.spectral_ctx).m_as(
                            uck.get("collision_coefficient")
                        ),
                        (-1, 1, 1),
                    ).astype(np.float32)
                )

            width = self.geometry.width.m_as(length_units)
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
                    "grid": Param(eval_albedo, ParamFlags.INIT),
                    "to_world": to_world,
                },
                "sigma_t": {
                    "type": "gridvolume",
                    "grid": Param(eval_sigma_t, ParamFlags.INIT),
                    "to_world": to_world,
                },
            }

        elif isinstance(self.geometry, SphericalShellGeometry):

            def eval_albedo(ctx):
                return mi.VolumeGrid(
                    np.reshape(
                        self.eval_albedo(ctx.spectral_ctx, None), (1, 1, -1)
                    ).astype(np.float32)
                )

            def eval_sigma_t(ctx):
                return mi.VolumeGrid(
                    np.reshape(
                        self.eval_sigma_t(ctx.spectral_ctx).m_as(
                            uck.get("collision_coefficient")
                        ),
                        (1, 1, -1),
                    ).astype(np.float32)
                )

            planet_radius = self.geometry.planet_radius.m_as(length_units)
            rmax = planet_radius + top
            to_world = mi.ScalarTransform4f.scale(rmax)

            volumes = {
                "albedo": {
                    "type": "sphericalcoordsvolume",
                    "volume": {
                        "type": "gridvolume",
                        "grid": Param(eval_albedo, ParamFlags.INIT),
                    },
                    "to_world": to_world,
                    "rmin": planet_radius / rmax,
                },
                "sigma_t": {
                    "type": "sphericalcoordsvolume",
                    "volume": {
                        "type": "gridvolume",
                        "grid": Param(eval_sigma_t, ParamFlags.INIT),
                    },
                    "to_world": to_world,
                    "rmin": planet_radius / rmax,
                },
            }

        else:  # Shouldn't happen, prevented by validator
            raise ValueError(
                f"unhandled atmosphere geometry type '{type(self.geometry).__name__}'"
            )

        # Create medium dictionary
        result = {
            "type": "heterogeneous",
            **volumes
            # Note: "phase" is deliberately unset, this is left to the
            # Atmosphere.template property
        }

        if self.scale is not None:
            result["scale"] = self.scale

        return result
