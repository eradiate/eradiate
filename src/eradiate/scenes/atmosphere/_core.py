from __future__ import annotations

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
from ..phase import PhaseFunction
from ..shapes import Shape
from ..._factory import Factory
from ...attrs import documented, get_doc, parse_docs
from ...contexts import KernelDictContext, SpectralContext
from ...kernel import (
    InitParameter,
    TypeIdLookupStrategy,
    UpdateParameter,
    map_unit_cube,
)
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
    Abstract base class defining common facilities for all atmospheres.

    An atmosphere expands as a :class:`mitsuba.PhaseFunction`, a
    :class:`mitsuba.Medium` and a :class:`mitsuba.Shape`.
    """

    id: str | None = documented(
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
        doc="Parameters defining the basic geometry of the scene.",
        type=".SceneGeometry",
        init_type=".SceneGeometry or dict or str, optional",
        default='"plane_parallel"',
    )

    # --------------------------------------------------------------------------
    #                               Properties
    # --------------------------------------------------------------------------

    @property
    def bottom_altitude(self) -> pint.Quantity:
        """
        Returns
        -------
        quantity
            Atmosphere bottom altitude.
        """
        return self.geometry.ground_altitude

    @property
    def top_altitude(self) -> pint.Quantity:
        """
        Returns
        -------
        quantity
            Atmosphere top altitude.
        """
        return self.geometry.toa_altitude

    @property
    def height(self) -> pint.Quantity:
        """
        Returns
        -------
        quantity
            Atmosphere height.
        """
        return self.top_altitude - self.bottom_altitude

    @property
    @abstractmethod
    def phase(self) -> PhaseFunction:
        """
        Returns
        -------
        .PhaseFunction
            Phase function associated with the atmosphere.
        """
        pass

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
    def shape(self) -> Shape:
        """
        Returns
        -------
        .Shape
            Shape associated with this atmosphere, based on the scene geometry.
        """
        return self.geometry.atmosphere_shape

    @property
    def shape_id(self):
        """
        Returns
        -------
        str
            ID of the shape associated with the atmosphere in the Mitsuba scene
            tree.
        """
        return f"shape_{self.id}"

    @property
    def medium_id(self):
        """
        Returns
        -------
        str
            ID of the medium associated with the atmosphere in the Mitsuba scene
            tree.
        """
        return f"medium_{self.id}"

    @property
    def phase_id(self):
        """
        Returns
        -------
        str
            ID of the phase function associated with the atmosphere in the
            Mitsuba scene tree.
        """
        return f"phase_{self.id}"

    @property
    @abstractmethod
    def _template_phase(self) -> dict:
        """
        Returns
        -------
        dict
            The phase function-related contribution to the kernel scene
            dictionary template for the atmosphere.
        """
        pass

    @property
    @abstractmethod
    def _template_medium(self) -> dict:
        """
        Returns
        -------
        dict
            The medium-related contribution to the kernel scene dictionary
            template for the atmosphere.
        """
        pass

    @property
    def _template_shape(self) -> dict:
        """
        Returns
        -------
        dict
            The shape-related contribution to the kernel scene dictionary
            template for the atmosphere.
        """
        template, _ = traverse(self.shape)
        return template.data

    @property
    def template(self) -> dict:
        # Inherit docstring
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
    def _params_phase(self) -> dict[str, UpdateParameter]:
        """
        Returns
        -------
        dict
            The phase function-related contribution to the parameter update map
            template for the atmosphere.
        """
        return {}

    @property
    def _params_medium(self) -> dict[str, UpdateParameter]:
        """
        Returns
        -------
        dict
            The medium-related contribution to the parameter update map template
            for the atmosphere.
        """
        return {}

    @property
    def _params_shape(self) -> dict[str, UpdateParameter]:
        """
        Returns
        -------
        dict
            The shape-related contribution to the parameter update map template
            for the atmosphere.
        """
        return {}

    @property
    def params(self) -> dict[str, UpdateParameter]:
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
    Abstract base class for heterogeneous atmospheres.
    """

    scale: float | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(float),
            validator=attrs.validators.optional(attrs.validators.instance_of(float)),
        ),
        doc="If set, the extinction coefficient is scaled by the corresponding "
        "amount during computation.",
        type="float or None",
        init_type="float, optional",
    )

    def update(self) -> None:
        """
        Update internal state.
        """
        pass

    # --------------------------------------------------------------------------
    #                    Spatial and thermophysical properties
    # --------------------------------------------------------------------------

    # Nothing at the moment

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    def eval_radprops(
        self,
        sctx: SpectralContext,
        zgrid: ZGrid | None = None,
        optional_fields: bool = False,
    ) -> xr.Dataset:
        """
        Evaluate the extinction coefficients and albedo profiles.

        Parameters
        ----------
        sctx : .SpectralContext
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        zgrid : .ZGrid, optional
            Altitude grid on which evaluation is performed. If unset, an
            instance-specific default is used
            (see :meth:`zgrid <.AbstractHeterogeneousAtmosphere.zgrid>`).

        optional_fields : bool, optional, default: False
            If ``True``, also output the absorption and scattering coefficients,
            not required for scene setup but useful for analysis and debugging.

        Returns
        -------
        Dataset
            A dataset with the following variables:

            * ``sigma_t``: extinction coefficient;
            * ``albedo``: albedo;
            * ``sigma_a``: absorption coefficient (optional);
            * ``sigma_s``: scattering coefficient (optional).

            and coordinates:

            * ``z``: altitude.
        """
        if zgrid is None:
            zgrid = self.geometry.zgrid

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
                        sigma_t.m_as(sigma_units) * (1.0 - albedo),
                        {
                            "units": f"{symbol(sigma_units)}",
                            "standard_name": "absorption_coefficient",
                            "long_name": "absorption coefficient",
                        },
                    ),
                    "sigma_s": (
                        "z_layer",
                        sigma_t.m_as(sigma_units) * albedo,
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
        self, sctx: SpectralContext, zgrid: ZGrid | None = None
    ) -> pint.Quantity:
        """
        Evaluate albedo spectrum based on a spectral context. This method
        dispatches evaluation to specialized methods depending on the active
        mode.

        Parameters
        ----------
        sctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        zgrid : .ZGrid, optional
            Altitude grid on which evaluation is performed. If unset, an
            instance-specific default is used
            (see :meth:`zgrid <.AbstractHeterogeneousAtmosphere.zgrid>`).

        Returns
        -------
        quantity
            Evaluated spectrum as an array with length equal to the number of
            layers.
        """

        pass

    @abstractmethod
    def eval_sigma_t(
        self, sctx: SpectralContext, zgrid: ZGrid | None = None
    ) -> pint.Quantity:
        """
        Evaluate extinction coefficient given a spectral context.

        Parameters
        ----------
        sctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        zgrid : .ZGrid, optional
            Altitude grid on which evaluation is performed. If unset, an
            instance-specific default is used
            (see :meth:`zgrid <.AbstractHeterogeneousAtmosphere.zgrid>`).

        Returns
        -------
        quantity
            Particle layer extinction coefficient.
        """

        pass

    @abstractmethod
    def eval_sigma_a(
        self, sctx: SpectralContext, zgrid: ZGrid | None = None
    ) -> pint.Quantity:
        """
        Evaluate absorption coefficient given a spectral context.

        Parameters
        ----------
        sctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        zgrid : .ZGrid, optional
            Altitude grid on which evaluation is performed. If unset, an
            instance-specific default is used
            (see :meth:`zgrid <.AbstractHeterogeneousAtmosphere.zgrid>`).

        Returns
        -------
        quantity
            Particle layer extinction coefficient.
        """

        pass

    @abstractmethod
    def eval_sigma_s(
        self, sctx: SpectralContext, zgrid: ZGrid | None = None
    ) -> pint.Quantity:
        """
        Evaluate scattering coefficient given a spectral context.

        Parameters
        ----------
        sctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        zgrid : .ZGrid, optional
            Altitude grid on which evaluation is performed. If unset, an
            instance-specific default is used
            (see :meth:`zgrid <.AbstractHeterogeneousAtmosphere.zgrid>`).

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
        # Inherit docstring
        length_units = uck.get("length")
        top = self.geometry.toa_altitude.m_as(length_units)
        bottom = self.geometry.ground_altitude.m_as(length_units)

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
                    "grid": InitParameter(
                        lambda ctx: mi.VolumeGrid(
                            np.reshape(
                                self.eval_albedo(ctx.spectral_ctx).m_as(
                                    ureg.dimensionless
                                ),
                                (-1, 1, 1),
                            ).astype(np.float32)
                        ),
                    ),
                    "to_world": to_world,
                },
                "sigma_t": {
                    "type": "gridvolume",
                    "grid": InitParameter(
                        lambda ctx: mi.VolumeGrid(
                            np.reshape(
                                self.eval_sigma_t(ctx.spectral_ctx).m_as(
                                    uck.get("collision_coefficient")
                                ),
                                (-1, 1, 1),
                            ).astype(np.float32)
                        ),
                    ),
                    "to_world": to_world,
                },
            }

        elif isinstance(self.geometry, SphericalShellGeometry):
            planet_radius = self.geometry.planet_radius.m_as(length_units)
            rmin = planet_radius
            rmax = rmin + top
            to_world = mi.ScalarTransform4f.scale(rmax)

            volumes = {
                "albedo": {
                    "type": "sphericalcoordsvolume",
                    "volume": {
                        "type": "gridvolume",
                        "grid": InitParameter(
                            lambda ctx: mi.VolumeGrid(
                                np.reshape(
                                    self.eval_albedo(ctx.spectral_ctx).m_as(
                                        ureg.dimensionless
                                    ),
                                    (1, 1, -1),
                                ).astype(np.float32)
                            ),
                        ),
                    },
                    "to_world": to_world,
                    "rmin": rmin / rmax,
                },
                "sigma_t": {
                    "type": "sphericalcoordsvolume",
                    "volume": {
                        "type": "gridvolume",
                        "grid": InitParameter(
                            lambda ctx: mi.VolumeGrid(
                                np.reshape(
                                    self.eval_sigma_t(ctx.spectral_ctx).m_as(
                                        uck.get("collision_coefficient")
                                    ),
                                    (1, 1, -1),
                                ).astype(np.float32)
                            ),
                        ),
                    },
                    "to_world": to_world,
                    "rmin": rmin / rmax,
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
    def _params_medium(self) -> dict[str, UpdateParameter]:
        # Inherit docstring
        if isinstance(self.geometry, PlaneParallelGeometry):
            return {
                "albedo.data": UpdateParameter(
                    lambda ctx: np.reshape(
                        self.eval_albedo(ctx.spectral_ctx).m_as(ureg.dimensionless),
                        (-1, 1, 1, 1),
                    ).astype(np.float32),
                    UpdateParameter.Flags.SPECTRAL,
                    lookup_strategy=TypeIdLookupStrategy(
                        node_type=mi.Medium,
                        node_id=self.medium_id,
                        parameter_relpath=f"albedo.data",
                    ),
                ),
                "sigma_t.data": UpdateParameter(
                    lambda ctx: np.reshape(
                        self.eval_sigma_t(ctx.spectral_ctx).m_as(
                            uck.get("collision_coefficient")
                        ),
                        (-1, 1, 1, 1),
                    ).astype(np.float32),
                    UpdateParameter.Flags.SPECTRAL,
                    lookup_strategy=TypeIdLookupStrategy(
                        node_type=mi.Medium,
                        node_id=self.medium_id,
                        parameter_relpath=f"sigma_t.data",
                    ),
                ),
            }

        elif isinstance(self.geometry, SphericalShellGeometry):
            return {
                "albedo.data": UpdateParameter(
                    lambda ctx: np.reshape(
                        self.eval_albedo(ctx.spectral_ctx).m_as(ureg.dimensionless),
                        (1, 1, -1, 1),
                    ).astype(np.float32),
                    UpdateParameter.Flags.SPECTRAL,
                    lookup_strategy=TypeIdLookupStrategy(
                        node_type=mi.Medium,
                        node_id=self.medium_id,
                        parameter_relpath=f"albedo.volume.data",
                    ),
                ),
                "sigma_t.data": UpdateParameter(
                    lambda ctx: np.reshape(
                        self.eval_sigma_t(ctx.spectral_ctx).m_as(
                            uck.get("collision_coefficient")
                        ),
                        (1, 1, -1, 1),
                    ).astype(np.float32),
                    UpdateParameter.Flags.SPECTRAL,
                    lookup_strategy=TypeIdLookupStrategy(
                        node_type=mi.Medium,
                        node_id=self.medium_id,
                        parameter_relpath=f"sigma_t.volume.data",
                    ),
                ),
            }

        else:  # Shouldn't happen, prevented by validator
            raise ValueError(
                f"unhandled atmosphere geometry type '{type(self.geometry).__name__}'"
            )
