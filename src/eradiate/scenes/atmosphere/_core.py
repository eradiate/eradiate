from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

import attrs
import mitsuba as mi
import numpy as np
import pint
import xarray as xr

from ..core import (
    CompositeSceneElement,
    SceneElement,
    traverse,
)
from ..geometry import PlaneParallelGeometry, SceneGeometry, SphericalShellGeometry
from ..shapes import CuboidShape, SphereShape
from ..._factory import Factory
from ...attrs import documented, get_doc, parse_docs
from ...contexts import KernelDictContext, SpectralContext
from ...kernel._kernel_dict import Parameter, ParamFlags
from ...kernel.transform import map_unit_cube
from ...radprops import ZGrid
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

    geometry: SceneGeometry = documented(
        attrs.field(
            default="plane_parallel",
            converter=SceneGeometry.convert,
            validator=attrs.validators.instance_of(SceneGeometry),
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
    def shape(self) -> t.Union[CuboidShape, SphereShape]:
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
    def shape_id(self):
        """
        str: Kernel dictionary key of the atmosphere's shape object.
        """
        return f"shape_{self.id}"

    @property
    def medium_id(self):
        """
        str: Kernel dictionary key of the atmosphere's medium object.
        """
        return f"medium_{self.id}"

    @property
    def phase_id(self):
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
        template, _ = traverse(self.shape)
        return template.data

    @property
    def template(self) -> dict:
        result = flatten(
            {
                self.phase_id: self._template_phase,
                self.medium_id: self._template_medium,
                self.shape_id: self._template_shape,
            }
        )

        result.update(
            {
                f"{self.medium_id}.phase.type": "ref",
                f"{self.medium_id}.phase.id": self.phase_id,
                f"{self.shape_id}.bsdf.type": "null",
                f"{self.shape_id}.interior.type": "ref",
                f"{self.shape_id}.interior.id": self.medium_id,
            }
        )

        return result

    @property
    def _params_phase(self) -> t.Dict[str, Parameter]:
        return {}

    @property
    def _params_medium(self) -> t.Dict[str, Parameter]:
        return {}

    @property
    def _params_shape(self) -> t.Dict[str, Parameter]:
        return {}

    @property
    def params(self) -> t.Dict[str, Parameter]:
        # Inherit docstring
        return flatten(
            {
                self.medium_id: {
                    **self._params_medium,
                    "phase_function": self._params_phase,
                },
                self.shape_id: self._params_shape,
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
        self,
        sctx: SpectralContext,
        zgrid: t.Optional[ZGrid] = None,
        optional_fields: bool = False,
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
        if zgrid is None:
            zgrid = self.zgrid

        sigma_units = ucc.get("collision_coefficient")
        sigma_t = self.eval_sigma_t(sctx, zgrid)
        albedo = self.eval_albedo(sctx, zgrid).m_as(ureg.dimensionless)

        data_vars = {
            "sigma_t": (
                "z_layer",
                sigma_t.m_as(sigma_units),
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
                        (sigma_t * (1.0 - albedo)).m_as(sigma_units),
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
                    zgrid.layers.magnitude,
                    {
                        "units": f"{symbol(zgrid.layers.units)}",
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
                    "grid": Parameter(
                        lambda ctx: mi.VolumeGrid(
                            np.reshape(
                                self.eval_albedo(ctx.spectral_ctx).m_as(
                                    ureg.dimensionless
                                ),
                                (-1, 1, 1),
                            ).astype(np.float32)
                        ),
                        ParamFlags.INIT,
                    ),
                    "to_world": to_world,
                },
                "sigma_t": {
                    "type": "gridvolume",
                    "grid": Parameter(
                        lambda ctx: mi.VolumeGrid(
                            np.reshape(
                                self.eval_sigma_t(ctx.spectral_ctx).m_as(
                                    uck.get("collision_coefficient")
                                ),
                                (-1, 1, 1),
                            ).astype(np.float32)
                        ),
                        ParamFlags.INIT,
                    ),
                    "to_world": to_world,
                },
            }

        elif isinstance(self.geometry, SphericalShellGeometry):
            planet_radius = self.geometry.planet_radius.m_as(length_units)
            rmax = planet_radius + top
            to_world = mi.ScalarTransform4f.scale(rmax)

            volumes = {
                "albedo": {
                    "type": "sphericalcoordsvolume",
                    "volume": {
                        "type": "gridvolume",
                        "grid": Parameter(
                            lambda ctx: mi.VolumeGrid(
                                np.reshape(
                                    self.eval_albedo(ctx.spectral_ctx).m_as(
                                        ureg.dimensionless
                                    ),
                                    (1, 1, -1),
                                ).astype(np.float32)
                            ),
                            ParamFlags.INIT,
                        ),
                    },
                    "to_world": to_world,
                    "rmin": planet_radius / rmax,
                },
                "sigma_t": {
                    "type": "sphericalcoordsvolume",
                    "volume": {
                        "type": "gridvolume",
                        "grid": Parameter(
                            lambda ctx: mi.VolumeGrid(
                                np.reshape(
                                    self.eval_sigma_t(ctx.spectral_ctx).m_as(
                                        uck.get("collision_coefficient")
                                    ),
                                    (1, 1, -1),
                                ).astype(np.float32)
                            ),
                            ParamFlags.INIT,
                        ),
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

    @property
    def _params_medium(self) -> t.Dict[str, Parameter]:
        if isinstance(self.geometry, PlaneParallelGeometry):
            return {
                "albedo.data": Parameter(
                    lambda ctx: np.reshape(
                        self.eval_albedo(ctx.spectral_ctx).m_as(ureg.dimensionless),
                        (-1, 1, 1, 1),
                    ).astype(np.float32),
                    ParamFlags.SPECTRAL,
                ),
                "sigma_t.data": Parameter(
                    lambda ctx: np.reshape(
                        self.eval_sigma_t(ctx.spectral_ctx).m_as(
                            uck.get("collision_coefficient")
                        ),
                        (-1, 1, 1, 1),
                    ).astype(np.float32),
                    ParamFlags.SPECTRAL,
                ),
            }

        elif isinstance(self.geometry, SphericalShellGeometry):
            return {
                "albedo.volume.data": Parameter(
                    lambda ctx: np.reshape(
                        self.eval_albedo(ctx.spectral_ctx).m_as(ureg.dimensionless),
                        (1, 1, -1, 1),
                    ).astype(np.float32),
                    ParamFlags.SPECTRAL,
                ),
                "sigma_t.volume.data": Parameter(
                    lambda ctx: np.reshape(
                        self.eval_sigma_t(ctx.spectral_ctx).m_as(
                            uck.get("collision_coefficient")
                        ),
                        (1, 1, -1, 1),
                    ).astype(np.float32),
                    ParamFlags.SPECTRAL,
                ),
            }

        else:  # Shouldn't happen, prevented by validator
            raise ValueError(
                f"unhandled atmosphere geometry type '{type(self.geometry).__name__}'"
            )
