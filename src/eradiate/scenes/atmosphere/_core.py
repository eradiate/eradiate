from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Literal

import attrs
import numpy as np
import pint
import xarray as xr
from axsdb import AbsorptionDatabase

from ..core import (
    CompositeSceneElement,
    SceneElement,
    traverse,
)
from ..geometry import (
    PlaneParallelGeometry,
    SceneGeometry,
)
from ..phase import PhaseFunction
from ..shapes import Shape
from ... import validators
from ..._factory import Factory
from ...attrs import define, documented, get_doc
from ...grid import SpatialGrid
from ...kernel import (
    SceneParameter,
)
from ...spectral.index import SpectralIndex
from ...units import symbol
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg
from ...util.deprecation import DeprecatedWarning
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
            "_merge.MergedComponent",
            "merged",
            {},
        ),
        (
            "_molecular.MolecularComponent",
            "molecular",
            {},
        ),
        (
            "_particle_component.ParticleOTComponent",
            "particle_ot",
            {"aliases": ["particle_layer"]},
        ),
        (
            "_particle_component.ParticleExtinctionComponent",
            "particle_extinction",
            {},
        ),
        (
            "_particle_state_component.ParticleStateComponent",
            "particle_state",
            {},
        ),
        (
            "_particle_state_component.ParticleMSComponent",
            "particle_ms",
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
            default=0.1,
            kw_only=True,
            converter=float,
            validator=[
                attrs.validators.instance_of(float),
                attrs.validators.ge(0.0),
                attrs.validators.le(1.0),
            ],
        ),
        doc="Specifies the probability to importance sample the phase using the "
        "emitter as incident direction. Set <= 0. to deactivate.",
        type="float",
    )

    rr_threshold: float = documented(
        attrs.field(
            default=0.1,
            kw_only=True,
            converter=float,
            validator=[
                attrs.validators.instance_of(float),
                attrs.validators.ge(0.0),
                attrs.validators.le(1.0),
            ],
        ),
        doc="Apply russian roulette to conditionally terminate the sample, "
        "when its transmittance is below this threshold. This helps increase "
        "performance at the cost of variance. Set to 0. to disable. Not "
        "applied on media that have the piecewise optimization.",
        type="float",
        init_type="float",
        default="0.1",
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


class _RadProps(ABC):
    """
    Radiative property evaluation interface shared by heterogeneous
    atmospheres and atmosphere components. Implementors must have a
    ``geometry`` attribute.
    """

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
        grid: SpatialGrid | None = None,
        optional_fields: bool = False,
    ) -> xr.Dataset:
        """
        Evaluate the extinction coefficients and albedo profiles.

        Parameters
        ----------
        si : .SpectralIndex
            Spectral index.

        grid : .SpatialGrid, optional
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

            with dimensions ``(x_cell, y_cell, z_layer)`` and coordinates:

            * ``x_cell``: X cell centres (azimuth for a spherical grid);
            * ``y_cell``: Y cell centres (colatitude for a spherical grid);
            * ``z_layer``: layer centre altitude.
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
                ("x_cell", "y_cell", "z_layer"),
                sigma_t.m_as(sigma_units),
                {
                    "units": f"{symbol(sigma_units)}",
                    "standard_name": "extinction_coefficient",
                    "long_name": "extinction coefficient",
                },
            ),
            "albedo": (
                ("x_cell", "y_cell", "z_layer"),
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
                        ("x_cell", "y_cell", "z_layer"),
                        sigma_t.m_as(sigma_units) * (1.0 - albedo),
                        {
                            "units": f"{symbol(sigma_units)}",
                            "standard_name": "absorption_coefficient",
                            "long_name": "absorption coefficient",
                        },
                    ),
                    "sigma_s": (
                        ("x_cell", "y_cell", "z_layer"),
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
                "x_cell": (
                    "x_cell",
                    grid.x_centers.magnitude,
                    {
                        "units": f"{symbol(grid.x_centers.units)}",
                        "standard_name": "cells_x",
                        "long_name": "cells x position",
                    },
                ),
                "y_cell": (
                    "y_cell",
                    grid.y_centers.magnitude,
                    {
                        "units": f"{symbol(grid.y_centers.units)}",
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
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        """
        Evaluate albedo spectrum based on a spectral context. This method
        dispatches evaluation to specialized methods depending on the active
        mode.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        grid : .SpatialGrid, optional
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
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        """
        Evaluate extinction coefficient given a spectral context.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        grid : .SpatialGrid, optional
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
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        """
        Evaluate absorption coefficient given a spectral context.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        grid : .SpatialGrid, optional
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
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        """
        Evaluate scattering coefficient given a spectral context.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        grid : .SpatialGrid, optional
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
            eval_sigma_func = eval_sigma[interaction]
        except KeyError as e:
            raise ValueError(
                f"invalid interaction type '{interaction}', "
                f"supported: {list(eval_sigma.keys())}"
            ) from e
        sigma = eval_sigma_func(si=si)
        dz = np.diff(self.geometry.grid.levels)
        tau = np.sum(np.multiply(sigma, dz).to("1"), axis=-1)
        return np.exp(-tau)

    def eval_transmittance_t(self, si: SpectralIndex) -> pint.Quantity:
        return self.eval_transmittance(si=si, interaction="extinction")

    def eval_transmittance_a(self, si: SpectralIndex) -> pint.Quantity:
        return self.eval_transmittance(si=si, interaction="absorption")

    def eval_transmittance_s(self, si: SpectralIndex) -> pint.Quantity:
        return self.eval_transmittance(si=si, interaction="scattering")


@define(eq=False, slots=False)
class AbstractHeterogeneousAtmosphere(Atmosphere, _RadProps, ABC):
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
        doc="If set to true, uses heterogeneous medium, which does not do "
        "piecewise optimization on 1D atmospheres. The piecewise optimization "
        "is only available for :class:`.PlaneParallelGeometry` with integrators "
        "that support piecewise traversal (e.g. :class:`.EOVolPathIntegrator`), "
        "other geometries use heterogeneous media.",
        type="bool",
        init_type="bool, optional",
    )

    extremum_resolution: tuple[int, int, int] = documented(
        attrs.field(
            default=(1, 1, 1),
            kw_only=True,
            converter=tuple,
            validator=validators.is_vector3,
        ),
        doc="[EXPERIMENTAL] Resolution of the extremum structure. Applies to "
        "both plane parallel and spherical shell geometries. Note that in "
        "plane parallel the dimensions are [x, y, z] and in spherical shell "
        "[r, theta, phi]. The default value (1, 1, 1) falls back to a global "
        "majorant. In plane parallel, set to (0,0,0) to trigger an adaptive "
        "search of the optimal resolution. Note that this is an expensive "
        "search that triggers at spectral update. Ignored by piecewise media "
        "(plane parallel with a 1D grid and ``force_majorant`` unset, see "
        "``is_piecewise``).",
        type="tuple[int, int, int]",
        init_type="tuple[int, int, int]",
        default="(1,1,1)",
    )

    def update(self) -> None:
        """
        Update internal state.
        """

    @property
    def is_piecewise(self) -> bool:
        """
        ``True`` if this atmosphere is rendered with a ``piecewise`` medium.
        Piecewise media get optimized during traversal when used with
        integrators that support it
        (see :attr:`.EOVolPathIntegrator.uses_eo_medium_api`).
        """
        return False


# Parameters which components accept for backward compatibility only
_DEPRECATED_COMPONENT_PARAMETERS = (
    "use_rrt",
    "ddis_threshold",
    "rr_threshold",
    "scale",
    "force_majorant",
)


def _deprecated_option(converter):
    return documented(
        attrs.field(
            default=None,
            kw_only=True,
            converter=attrs.converters.optional(converter),
            repr=False,
        ),
        doc="Deprecated, set it on :class:`.HeterogeneousAtmosphere` instead.",
        type="None",
        default="None",
    )


@define(eq=False, slots=False)
class AtmosphereComponent(_RadProps, ABC):
    """
    Abstract base class for atmosphere components.

    Components provide radiative properties and a phase function. They are
    assembled by a :class:`.HeterogeneousAtmosphere`, which generates the
    corresponding kernel scene dictionary.
    """

    id: str | None = documented(
        attrs.field(
            default="atmosphere",
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc="Component identifier. Overridden by the parent "
        ":class:`.HeterogeneousAtmosphere`.",
        type="str or None",
        init_type="str, optional",
        default='"atmosphere"',
    )

    geometry: SceneGeometry | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(SceneGeometry.convert),
            validator=attrs.validators.optional(
                attrs.validators.instance_of(SceneGeometry)
            ),
        ),
        doc="Scene geometry on which radiative properties are evaluated. If "
        "unset, a default plane parallel geometry is used, and the parent "
        ":class:`.HeterogeneousAtmosphere` overrides it with the geometry of "
        "the merged component it belongs to. If set, it is never overridden "
        "and is used to group components "
        "(see :attr:`.HeterogeneousAtmosphere.merge_policy`). In all cases, "
        "radiative properties are rendered on the grid of the merged "
        "component.",
        type=".SceneGeometry",
        init_type=".SceneGeometry or dict or str, optional",
        default="None",
    )

    extremum_resolution: tuple[int, int, int] | None = documented(
        attrs.field(
            default=None,
            kw_only=True,
            converter=attrs.converters.optional(tuple),
            validator=attrs.validators.optional(validators.is_vector3),
        ),
        doc="[EXPERIMENTAL] Resolution of the extremum structure of the medium "
        "generated for this component when it is not merged with other "
        "components. If unset, the resolution of the parent "
        ":class:`.HeterogeneousAtmosphere` is used.",
        type="tuple[int, int, int] or None",
        init_type="tuple[int, int, int], optional",
        default="None",
    )

    # Deprecated Parameters
    use_rrt: bool | None = _deprecated_option(bool)
    ddis_threshold: float | None = _deprecated_option(float)
    rr_threshold: float | None = _deprecated_option(float)
    scale: float | None = _deprecated_option(float)
    force_majorant: bool | None = _deprecated_option(bool)

    def __attrs_post_init__(self):
        self._own_geometry = self.geometry
        if self.geometry is None:
            self.geometry = PlaneParallelGeometry()

        deprecated_options = [
            f"'{name}'"
            for name in _DEPRECATED_COMPONENT_PARAMETERS
            if getattr(self, name) is not None
        ]

        if deprecated_options:
            deprecated_list = ",".join(deprecated_options)
            warnings.warn(
                DeprecatedWarning(
                    component=f"Setting {{{deprecated_list}}} on '{type(self).__name__}'",
                    deprecated_in="1.3.1",
                    removed_in="2.0.0",
                    details="It will have no effect and will be ignored, set "
                    "them on 'HeterogeneousAtmosphere' instead.",
                ),
                stacklevel=3,
            )

        self.update()

    def update(self) -> None:
        """
        Update internal state.
        """

    @property
    def phase_id(self) -> str:
        """
        Returns
        -------
        str
            ID of the phase function associated with the component.
        """
        return f"phase_{self.id}"

    @property
    @abstractmethod
    def phase(self) -> PhaseFunction:
        """
        Returns
        -------
        .PhaseFunction
            Phase function associated with the component.
        """
