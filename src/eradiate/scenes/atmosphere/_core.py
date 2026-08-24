from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import attrs
import mitsuba as mi
import numpy as np
import pint
import xarray as xr
from axsdb import AbsorptionDatabase

from eradiate.kernel._render import SearchSceneParameter

from .._gridvolume import generate_gridvolume
from ..core import (
    CompositeSceneElement,
    SceneElement,
    traverse,
)
from ..geometry import (
    PlaneParallelGeometry,
    SceneGeometry,
    SphericalShellGeometry,
)
from ..phase import PhaseFunction
from ..shapes import Shape
from ... import validators
from ..._factory import Factory
from ..._mode import get_mode
from ...attrs import define, documented, get_doc
from ...contexts import KernelContext
from ...grid import GridCoords
from ...kernel import (
    KernelSceneParameterFlags,
    SceneParameter,
)
from ...spectral.index import SpectralIndex
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
            "_molecular.MolecularAtmosphere",
            "molecular",
            {},
        ),
        (
            "_particle_ensemble.ParticleEnsemble",
            "particle_ensemble",
            {},
        ),
    ],
    cls_prefix="eradiate.scenes.atmosphere",
)


@define(eq=False, slots=False)
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
        doc="Parameters defining the basic geometry of the scene. "
        "Note if the atmosphere is used in a simulation, the experiment has "
        "all control over the atmosphere's geometry and is going to set it. "
        "Therefore, in such a case it is best to define the geometry at the "
        "experiment level. When the atmosphere is used standalone, care must "
        "be taken to ensure that the geometry is consistent with the vertical "
        "extent of the atmosphere.",
        type=".SceneGeometry",
        init_type=".SceneGeometry or dict or str, optional",
        default='"plane_parallel"',
    )

    use_rrt: bool = documented(
        attrs.field(
            default=True,
            kw_only=True,
            converter=bool,
            validator=attrs.validators.instance_of(bool),
        ),
        doc="If set to true, prescribe the use of residual ratio tracking to "
        "estimate transmittance for integrators that allow it. "
        "Otherwise, falls back to ratio tracking. "
        "Residual ratio tracking generally reduces variance and can improve "
        "performance for high optical thickness.",
        type="bool",
    )

    ddis_threshold: float = documented(
        attrs.field(
            default=0.0,
            kw_only=True,
            converter=float,
            validator=attrs.validators.instance_of(float),
        ),
        doc="Specifies the probability to importance sample the phase using the "
        "emitter as incident direction. Set negative to deactivate.",
        type="float",
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

    # --------------------------------------------------------------------------
    #                    Spatial and thermophysical properties
    # --------------------------------------------------------------------------

    @abstractmethod
    def eval_mfp(self, ctx: KernelContext) -> pint.Quantity:
        """
        Compute a typical scattering mean free path. This rough estimate can be
        used *e.g.* to compute a distance guaranteeing that the medium can be
        considered optically thick.

        Parameters
        ----------
        ctx : :class:`.KernelContext`
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns
        -------
        mfp : quantity
            Mean free path estimate.
        """

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
    @abstractmethod
    def _params_phase(self) -> dict[str, SceneParameter]:
        """
        Returns
        -------
        dict
            The phase function-related contribution to the parameter update map
            template for the atmosphere.
        """

    @property
    @abstractmethod
    def _params_medium(self) -> dict[str, SceneParameter]:
        """
        Returns
        -------
        dict
            The medium-related contribution to the parameter update map template
            for the atmosphere.
        """

    @property
    def _params_shape(self) -> dict[str, SceneParameter]:
        """
        Returns
        -------
        dict
            The shape-related contribution to the parameter update map template
            for the atmosphere.
        """
        _, umap = traverse(self.shape)
        return umap.data

    @property
    def params(self) -> dict[str, SceneParameter]:
        # Inherit docstring
        return flatten(
            {
                self.phase_id: self._params_phase,
                self.medium_id: self._params_medium,
                self.shape_id: self._params_shape,
            }
        )


@define(eq=False, slots=False)
class AbstractHeterogeneousAtmosphere(Atmosphere, ABC):
    """
    Abstract base class for heterogeneous atmospheres.
    """

    scale: float | None = documented(
        attrs.field(
            default=None,
            kw_only=True,
            converter=attrs.converters.optional(float),
            validator=attrs.validators.optional(attrs.validators.instance_of(float)),
        ),
        doc="If set, the extinction coefficient is scaled by the corresponding "
        "amount during computation.",
        type="float or None",
        init_type="float, optional",
    )

    force_majorant: bool = documented(
        attrs.field(
            default=False,
            kw_only=True,
            converter=attrs.converters.optional(bool),
            validator=attrs.validators.optional(attrs.validators.instance_of(bool)),
        ),
        doc="If set to true, uses heterogeneous medium, which is compatible with all "
        "integrators except PiecewiseVolpathIntegrator. Otherwise, uses a piecewise medium which "
        "is compatible with PiecewiseVolPathIntegrator and other integrators. This setting "
        "only affects PlaneParallelGeometry, other geometries use heterogeneous mediums.",
        type="bool",
        init_type="bool, optional",
    )

    extremum_resolution: tuple[int, int, int] = documented(
        attrs.field(
            default=(1, 1, 1),
            validator=validators.is_vector3,
        ),
        doc="[EXPERIMENTAL] Resolution of the extremum structure. Applies to "
        "both plane parallel and spherical shell geometries but the latter only "
        "accepts a resolution greater than one along its radial dimension "
        "(first dimension). Also note than in plane parallel the dimensions are "
        "[x, y, z] and in spherical shell [r, theta, phi]. "
        "The default value (1, 1, 1) falls back to a global majorant. In plane "
        "parallel, set to (0,0,0) to trigger an adaptive search of the optimal"
        "resolution. Note that this is an expensive search that triggers at "
        "spectral update.",
        type="tuple[int, int, int]",
        init_type="tuple[int, int, int]",
        default="(1,1,1)",
    )

    def update(self) -> None:
        """
        Update internal state.
        """

    # --------------------------------------------------------------------------
    #                    Spatial and thermophysical properties
    # --------------------------------------------------------------------------

    # Nothing at the moment

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    @property
    def absorption_data(self) -> AbsorptionDatabase | None:
        """
        Returns
        -------
        .AbsorptionDatabase or None
            If relevant, the molecular absorption database associated with this
            atmosphere.
        """
        return None

    def eval_radprops(
        self,
        si: SpectralIndex,
        grid: GridCoords | None = None,
        optional_fields: bool = False,
    ) -> xr.Dataset:
        """
        Evaluate the extinction coefficients and albedo profiles.

        Parameters
        ----------
        si : .SpectralIndex
            Spectral index.

        grid : .GridCoords, optional
            Altitude grid on which evaluation is performed. If unset, an
            instance-specific default is used
            (see :meth:`grid <.AbstractHeterogeneousAtmosphere.geometry.grid>`).

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
        if grid is None:
            grid = self.geometry.grid

        sigma_units = ucc.get("collision_coefficient")
        sigma_t = self.eval_sigma_t(si, grid)
        sigma_t = np.broadcast_to(sigma_t, grid.shape)
        albedo = self.eval_albedo(si, grid).m_as(ureg.dimensionless)
        albedo = np.broadcast_to(albedo, grid.shape)

        data_vars = {
            "sigma_t": (
                ("x_cells", "y_cells", "z_layer"),
                sigma_t.m_as(sigma_units),
                {
                    "units": f"{symbol(sigma_units)}",
                    "standard_name": "extinction_coefficient",
                    "long_name": "extinction coefficient",
                },
            ),
            "albedo": (
                ("x_cells", "y_cells", "z_layer"),
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
                        ("x_cells", "y_cells", "z_layer"),
                        sigma_t.m_as(sigma_units) * (1.0 - albedo),
                        {
                            "units": f"{symbol(sigma_units)}",
                            "standard_name": "absorption_coefficient",
                            "long_name": "absorption coefficient",
                        },
                    ),
                    "sigma_s": (
                        ("x_cells", "y_cells", "z_layer"),
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
                "x_layer": (
                    "x_layer",
                    grid.cells_x.magnitude,
                    {
                        "units": f"{symbol(grid.cells_x.units)}",
                        "standard_name": "cells_x",
                        "long_name": "cells x position",
                    },
                ),
                "y_layer": (
                    "y_layer",
                    grid.cells_y.magnitude,
                    {
                        "units": f"{symbol(grid.layers.units)}",
                        "standard_name": "cells_y",
                        "long_name": "cells y position",
                    },
                ),
                "z_layer": (
                    "z_layer",
                    grid.layers.magnitude,
                    {
                        "units": f"{symbol(grid.layers.units)}",
                        "standard_name": "layer_altitude",
                        "long_name": "layer altitude",
                    },
                ),
            },
        )

    @abstractmethod
    def eval_albedo(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> pint.Quantity:
        """
        Evaluate albedo spectrum based on a spectral context. This method
        dispatches evaluation to specialized methods depending on the active
        mode.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        grid : .GridCoords, optional
            Altitude grid on which evaluation is performed. If unset, an
            instance-specific default is used
            (see :meth:`grid <.AbstractHeterogeneousAtmosphere.geometry.grid>`).

        Returns
        -------
        quantity
            Evaluated spectrum as an array with length equal to the number of
            layers.
        """

    @abstractmethod
    def eval_sigma_t(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> pint.Quantity:
        """
        Evaluate extinction coefficient given a spectral context.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        grid : .GridCoords, optional
            Altitude grid on which evaluation is performed. If unset, an
            instance-specific default is used
            (see :meth:`grid <.AbstractHeterogeneousAtmosphere.geometry.grid>`).

        Returns
        -------
        quantity
            Extinction coefficient.
        """

    @abstractmethod
    def eval_sigma_a(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> pint.Quantity:
        """
        Evaluate absorption coefficient given a spectral context.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        grid : .GridCoords, optional
            Altitude grid on which evaluation is performed. If unset, an
            instance-specific default is used
            (see :meth:`grid <.AbstractHeterogeneousAtmosphere.geometry.grid>`).

        Returns
        -------
        quantity
            Absorption coefficient.
        """

    @abstractmethod
    def eval_sigma_s(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> pint.Quantity:
        """
        Evaluate scattering coefficient given a spectral context.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        grid : .GridCoords, optional
            Altitude grid on which evaluation is performed. If unset, an
            instance-specific default is used
            (see :meth:`grid <.AbstractHeterogeneousAtmosphere.geometry.grid>`).

        Returns
        -------
        quantity
            Scattering coefficient.
        """

    def eval_transmittance(
        self,
        si: SpectralIndex,
        interaction: Literal["extinction", "absorption", "scattering"] = "extinction",
    ) -> pint.Quantity:
        """
        Evaluate the atmosphere's transmittance with respect to specific
        interaction.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        interaction : {"extinction", "absorption", "scattering"}, optional, default: "extinction"
            Interaction type.

        Returns
        -------
        quantity
            Total atmosphere transmittance.
        """
        eval_sigma = {
            "extinction": self.eval_sigma_t,
            "absorption": self.eval_sigma_a,
            "scattering": self.eval_sigma_s,
        }
        try:
            sigma = eval_sigma[interaction](si=si)
        except KeyError as e:
            raise TypeError(
                f"invalid interaction type '{interaction}', "
                f"supported: {list(eval_sigma.keys())}"
            ) from e
        dz = np.diff(self.geometry.grid.levels)
        tau = np.sum(np.multiply(sigma, dz).to("1"), axis=-1)
        return np.exp(-tau)

    def eval_transmittance_t(self, si: SpectralIndex) -> pint.Quantity:
        return self.eval_transmittance(si=si, interaction="extinction")

    def eval_transmittance_a(self, si: SpectralIndex) -> pint.Quantity:
        return self.eval_transmittance(si=si, interaction="absorption")

    def eval_transmittance_s(self, si: SpectralIndex) -> pint.Quantity:
        return self.eval_transmittance(si=si, interaction="scattering")

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def _template_medium(self) -> dict:
        # Inherit docstring

        extremum = None

        volumes = {
            "albedo": generate_gridvolume(
                self.geometry,
                self.eval_albedo,
                units=ureg.dimensionless,
                dtype=np.float64,
            ),
            "sigma_t": generate_gridvolume(
                self.geometry,
                self.eval_sigma_t,
                units=uck.get("collision_coefficient"),
                dtype=np.float64,
            ),
        }
        sigma_t_id = f"{self.id}_sigma_t"

        piecewise = (
            isinstance(self.geometry, PlaneParallelGeometry)
            and not self.force_majorant
            and self.geometry.grid.onedim
        )

        aabb_conf = {}
        if isinstance(self.geometry, SphericalShellGeometry):
            extr_conf = {
                "type": "extremum_spherical",
                "rmin": self.geometry.atmosphere_volume_rmin,
            }
        elif not piecewise:
            extr_conf = {"type": "extremum_grid"}
            aabb_min = self.geometry.bbox.min.m_as("m")
            # The geometry bbox inserts a sub-surface padding. We want the data to be evaluated
            # on the entire geometry extent in x and y directions, but we should avoid the wrap_mode
            # to affect the z direction.
            aabb_min[2] = self.geometry.grid.levels[0].m_as("m")
            aabb_conf = {
                "aabb_min": aabb_min,
                "aabb_max": self.geometry.bbox.max.m_as("m"),
            }

        if piecewise:
            medium = "piecewise"
        else:
            if self.extremum_resolution != (1, 1, 1):
                extremum = {
                    **extr_conf,
                    "volume": {"type": "ref", "id": sigma_t_id},
                    "resolution": self.extremum_resolution,
                    "to_world": self.geometry.grid.to_world,
                    "wrap_mode": str(self.geometry.wrap_mode).lower(),
                }
            medium = "eoheterogeneous"

        # Create medium dictionary
        result = {
            "type": medium,
            "has_spectral_extinction": (
                # piecewise volpath currently only works with spectral extinction true
                # hack fix by setting to True when we have a piecewise medium
                (not get_mode().check(mi_color_mode="mono")) or (medium == "piecewise")
            ),
            **aabb_conf,
            **volumes,
            # Note: "phase" is deliberately unset, this is left to the
            # Atmosphere.template property
        }

        if self.scale is not None:
            result["scale"] = self.scale

        if extremum is not None:
            result["extremum"] = extremum
            result["sigma_t"]["id"] = sigma_t_id

        if medium == "eoheterogeneous":
            result["use_rrt"] = self.use_rrt

        result["ddis_threshold"] = self.ddis_threshold

        return result

    @property
    def _params_medium(self) -> dict[str, SceneParameter]:
        # Inherit docstring

        if isinstance(self.geometry, PlaneParallelGeometry):
            albedo_key = "albedo.data"
            sigma_t_key = "sigma_t.data"
        elif isinstance(self.geometry, SphericalShellGeometry):
            albedo_key = "albedo.volume.data"
            sigma_t_key = "sigma_t.volume.data"
        else:
            raise TypeError(
                f"unhandled scene geometry type '{type(self.geometry).__name__}'"
            )

        albedo_search = SearchSceneParameter(
            node_type=mi.Medium, node_id=self.medium_id, parameter_relpath=albedo_key
        )
        sigma_t_search = SearchSceneParameter(
            node_type=mi.Medium, node_id=self.medium_id, parameter_relpath=sigma_t_key
        )
        albedo_grid = generate_gridvolume(
            self.geometry,
            self.eval_albedo,
            units=ureg.dimensionless,
            search=albedo_search,
            flag=KernelSceneParameterFlags.SPECTRAL,
            dtype=np.float64,
        )
        sigma_t_grid = generate_gridvolume(
            self.geometry,
            self.eval_sigma_t,
            units=uck.get("collision_coefficient"),
            search=sigma_t_search,
            flag=KernelSceneParameterFlags.SPECTRAL,
            dtype=np.float64,
        )

        return {
            albedo_key: albedo_grid,
            sigma_t_key: sigma_t_grid,
        }
