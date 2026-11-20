"""
Atmosphere component container.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import MutableMapping

import attrs
import mitsuba as mi
import numpy as np
import pint
from axsdb import AbsorptionDatabase

from ._core import (
    AtmosphereComponent,
    AtmosphereComponentFlag,
    EvalRadProps,
    atmosphere_factory,
)
from ._molecular import MolecularComponent
from ._particle_component import ParticleComponent
from .._gridvolume import generate_gridvolume
from ..core import (
    CompositeSceneElement,
    SceneElement,
    traverse,
)
from ..geometry import (
    FilterType,
    PlaneParallelGeometry,
    SceneGeometry,
    SphericalShellGeometry,
)
from ..phase import MultiPhaseFunction, PhaseFunction
from ..shapes import Shape
from ... import validators
from ...attrs import define, documented, get_doc
from ...grid import SpatialGrid
from ...kernel import (
    DictParameter,
    KernelSceneParameterFlags,
    SceneParameter,
    SearchSceneParameter,
)
from ...spectral.index import SpectralIndex
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg
from ...util.deprecation import DeprecatedWarning
from ...util.misc import cache_by_id, flatten

logger = logging.getLogger(__name__)


def _uniform_value(value: pint.Quantity, units: pint.Unit) -> float:
    """
    Magnitude of a spatially uniform quantity, which may be a scalar or an
    array of identical values.
    """
    return float(np.ravel(value.m_as(units))[0])


def _components_converter(value):
    if not value:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [
        atmosphere_factory.convert(element, allowed_cls=AtmosphereComponent)
        for element in value
    ]


@define(eq=False, slots=False)
class Atmosphere(CompositeSceneElement, EvalRadProps):
    """
    Atmosphere scene element [``atmosphere``].

    An atmosphere assembles a list of :class:`.AtmosphereComponent` into a
    single medium. It expands as a :class:`mitsuba.PhaseFunction`, a
    :class:`mitsuba.Medium` and a :class:`mitsuba.Shape`. Both 1D
    (layer-based) and 3D (fully gridded) configurations are supported.
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

    components: list[AtmosphereComponent] = documented(
        attrs.field(
            factory=list,
            converter=_components_converter,
            validator=attrs.validators.deep_iterable(
                attrs.validators.instance_of(AtmosphereComponent)
            ),
        ),
        doc="List of atmosphere components, *e.g.* a "
        ":class:`.HomogeneousComponent`, a :class:`.MolecularComponent` or a "
        ":class:`.ParticleComponent`. Elements may be specified as dictionaries "
        "interpreted by :data:`.atmosphere_factory`; in that case, the "
        "``type`` parameter is required. At most one molecular component is "
        "allowed.",
        type="list of .AtmosphereComponent",
        init_type="list of .AtmosphereComponent or dict, optional",
        default="[]",
    )

    @components.validator
    def _components_validator(self, attribute, value):
        if sum(AtmosphereComponentFlag.MOLECULAR in c.flags for c in value) > 1:
            raise ValueError(
                f"while validating {attribute.name}: at most one "
                "molecular component is allowed"
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
        init_type="bool",
        default="False",
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
        "``is_piecewise``) and by homogeneous atmospheres (see "
        "``is_homogeneous``).",
        type="tuple[int, int, int]",
        init_type="tuple[int, int, int]",
        default="(1,1,1)",
    )

    use_phase_mis: bool = documented(
        attrs.field(default=True, converter=bool, kw_only=True),
        doc="If ``True``, multiple importance sampling is enabled for the "
        "3D phase function mixture.",
        type="bool",
        init_type="bool",
        default="True",
    )

    def __init__(
        self,
        *args,
        molecular_atmosphere=None,
        particle_components=None,
        particle_layers=None,
        **kwargs,
    ):
        legacy = {
            name: value
            for name, value in [
                ("molecular_atmosphere", molecular_atmosphere),
                ("particle_components", particle_components),
                ("particle_layers", particle_layers),
            ]
            if value is not None
        }

        if legacy:
            if "components" in kwargs:
                raise TypeError(
                    "cannot pass 'components' together with "
                    "'molecular_atmosphere', 'particle_components' or "
                    "'particle_layers'"
                )

            for name in legacy:
                warnings.warn(
                    DeprecatedWarning(
                        component=f"'Atmosphere.{name}'",
                        deprecated_in="1.4.0",
                        removed_in="2.0.0",
                        details="Use 'components' instead.",
                    ),
                    stacklevel=3,
                )

            # Molecular first in order to match legacy input
            components = []
            if molecular_atmosphere is not None:
                if (
                    isinstance(molecular_atmosphere, MutableMapping)
                    and "type" not in molecular_atmosphere
                ):
                    molecular_atmosphere = {**molecular_atmosphere, "type": "molecular"}
                components.append(
                    atmosphere_factory.convert(
                        molecular_atmosphere, allowed_cls=MolecularComponent
                    )
                )
            for particles in [particle_components, particle_layers]:
                if particles:
                    if not isinstance(particles, (list, tuple)):
                        particles = [particles]
                    components += [ParticleComponent.convert(c) for c in particles]

            kwargs["components"] = components

        self.__attrs_init__(*args, **kwargs)

    # --------------------------------------------------------------------------
    #                               Properties
    # --------------------------------------------------------------------------

    @property
    def molecular_component(self) -> AtmosphereComponent | None:
        """
        Molecular component of this atmosphere, if any.
        """
        molecular = [
            c for c in self.components if AtmosphereComponentFlag.MOLECULAR in c.flags
        ]
        return molecular[0] if len(molecular) > 0 else None

    @property
    def molecular_atmosphere(self) -> AtmosphereComponent | None:
        """
        Molecular component of this atmosphere, if any.

        .. deprecated:: 1.4.0
           Use :attr:`molecular_component` instead.
        """
        warnings.warn(
            DeprecatedWarning(
                component="'Atmosphere.molecular_atmosphere'",
                deprecated_in="1.4.0",
                removed_in="2.0.0",
                details="Use 'molecular_component' instead.",
            ),
            stacklevel=2,
        )
        return self.molecular_component

    @property
    def particle_components(self) -> list[AtmosphereComponent]:
        """
        Particle components of this atmosphere, if any.
        """
        return [
            c for c in self.components if AtmosphereComponentFlag.PARTICLE in c.flags
        ]

    @property
    def particle_layers(self) -> list[AtmosphereComponent]:
        """
        Particle components of this atmosphere.

        .. deprecated:: 1.3.0
           Use :attr:`particle_components` instead.
        """
        warnings.warn(
            DeprecatedWarning(
                component="'Atmosphere.particle_layers'",
                deprecated_in="1.3.0",
                removed_in="2.0.0",
                details="Use 'particle_components' instead.",
            ),
            stacklevel=2,
        )
        return self.particle_components

    @property
    def is_homogeneous(self) -> bool:
        """
        ``True`` if all components are homogeneous. The atmosphere is then
        rendered with a ``homogeneous`` medium and extremum structures are
        ignored.
        """
        return all(
            AtmosphereComponentFlag.HOMOGENEOUS in c.flags for c in self.components
        )

    @property
    def is_piecewise(self) -> bool:
        """
        ``True`` if this atmosphere is rendered with a ``piecewise`` medium,
        which requires a :class:`.PlaneParallelGeometry` with a 1D grid, the
        ``nearest`` filter type, and ``force_majorant`` unset. Piecewise media
        get optimized during traversal when used with integrators that support
        it (see :attr:`.EOVolPathIntegrator.uses_eo_medium_api`).
        """
        return (
            not self.is_homogeneous
            and isinstance(self.geometry, PlaneParallelGeometry)
            and not self.force_majorant
            and self.geometry.grid.is_1d
            and self.geometry.filter_type is FilterType.NEAREST
        )

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

    def update(self) -> None:
        """
        Update internal state: force component IDs and geometries.
        """
        if not self.components:
            raise ValueError("Atmosphere must have at least one component")

        for i, component in enumerate(self.components):
            component.update()
            component.id = f"{self.id}_component_{i}"
            logger.debug(
                "Override the geometry of component '%s' to %s (was %s)",
                component.id,
                self.geometry,
                component.geometry,
            )
            component.geometry = self.geometry

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    @property
    def absorption_data(self) -> AbsorptionDatabase | None:
        # Inherit docstring
        if self.molecular_component is not None:
            return self.molecular_component.absorption_data
        else:
            return None

    def eval_albedo(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        if grid is not None and grid is not self.geometry.grid:
            raise ValueError("grid must be left unset or set to self.geometry.grid")

        units = ucc.get("collision_coefficient")
        sigma_s = self.eval_sigma_s(si).m_as(units)
        sigma_t = self.eval_sigma_t(si).m_as(units)
        albedo = np.zeros_like(sigma_s)
        np.divide(sigma_s, sigma_t, where=sigma_t != 0.0, out=albedo)

        return albedo * ureg.dimensionless

    @cache_by_id
    def _eval_sigma_t_impl(self, si: SpectralIndex) -> pint.Quantity:
        sigma_units = ucc.get("collision_coefficient")
        result = []
        for component in self.components:
            cmp_result = component.eval_sigma_t(si, self.geometry.grid)
            result.append(
                np.broadcast_to(cmp_result.m_as(sigma_units), self.geometry.grid.shape)
            )
        return np.stack(result) * sigma_units

    def eval_sigma_t(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        if grid is not None and grid is not self.geometry.grid:
            raise ValueError("grid must be left unset or set to self.geometry.grid")
        return self._eval_sigma_t_impl(si).sum(axis=0)

    def eval_sigma_a(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        if grid is not None and grid is not self.geometry.grid:
            raise ValueError("grid must be left unset or set to self.geometry.grid")
        return self.eval_sigma_t(si) - self.eval_sigma_s(si)

    @cache_by_id
    def _eval_sigma_s_impl(self, si: SpectralIndex) -> pint.Quantity:
        sigma_units = ucc.get("collision_coefficient")
        result = []
        for component in self.components:
            cmp_result = component.eval_sigma_s(si, self.geometry.grid)
            result.append(
                np.broadcast_to(cmp_result.m_as(sigma_units), self.geometry.grid.shape)
            )
        return np.stack(result) * sigma_units

    def _eval_sigma_s_component(
        self, si: SpectralIndex, n_component: int
    ) -> pint.Quantity:
        return self._eval_sigma_s_impl(si)[n_component]

    def eval_sigma_s(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        if grid is not None and grid is not self.geometry.grid:
            raise ValueError("grid must be left unset or set to self.geometry.grid")
        return self._eval_sigma_s_impl(si).sum(axis=0)

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def phase(self) -> PhaseFunction:
        """
        Returns
        -------
        .PhaseFunction
            Phase function associated with the atmosphere: the phase function
            of the single component, or a :class:`.MultiPhaseFunction` mixing
            the component phase functions weighted by their scattering
            coefficients.
        """
        if len(self.components) == 1:
            return self.components[0].phase

        components, weights = [], []
        sigma_units = ucc.get("collision_coefficient")

        for i, component in enumerate(self.components):
            components.append(component.phase)

            def eval_sigma_s(si: SpectralIndex, n_component: int = i) -> np.ndarray:
                return self._eval_sigma_s_component(si, n_component).m_as(sigma_units)

            weights.append(eval_sigma_s)

        return MultiPhaseFunction(
            components=components,
            weights=weights,
            geometry=self.geometry,
            use_mis=self.use_phase_mis,
        )

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
    def shape_id(self) -> str:
        """
        Returns
        -------
        str
            ID of the shape associated with the atmosphere in the Mitsuba scene
            tree.
        """
        return f"shape_{self.id}"

    @property
    def medium_id(self) -> str:
        """
        Returns
        -------
        str
            ID of the medium associated with the atmosphere in the Mitsuba scene
            tree.
        """
        return f"medium_{self.id}"

    @property
    def phase_id(self) -> str:
        """
        Returns
        -------
        str
            ID of the phase function associated with the atmosphere in the
            Mitsuba scene tree.
        """
        return f"phase_{self.id}"

    def _build_medium_template(self, source: EvalRadProps) -> dict:
        """
        Generate a medium template from the radiative properties of ``source``,
        evaluated on the geometry of ``source``. Medium options are taken from
        this atmosphere.

        Parameters
        ----------
        source : .Atmosphere or .AtmosphereComponent
            Radiative property source, *e.g.* the atmosphere itself or one of
            its components.

        Returns
        -------
        dict
            Medium template.
        """
        collision_coefficient_units = uck.get("collision_coefficient")
        length_units = uck.get("length")

        result = {
            "has_spectral_extinction": False,
            "rr_threshold": self.rr_threshold,
            "use_rrt": self.use_rrt,
            "ddis_threshold": self.ddis_threshold,
            # Note: "phase" is deliberately unset, this is left to the
            # Atmosphere.template property
        }
        if self.scale is not None:
            result["scale"] = self.scale

        if self.is_homogeneous:
            # Grid lookups at each collision slow down spherical shell
            # rendering, while a constant volume does not
            return {
                **result,
                "type": "homogeneous",
                "albedo": DictParameter(
                    lambda ctx: _uniform_value(
                        source.eval_albedo(ctx.si), ureg.dimensionless
                    )
                ),
                "sigma_t": DictParameter(
                    lambda ctx: _uniform_value(
                        source.eval_sigma_t(ctx.si), collision_coefficient_units
                    )
                ),
            }

        geometry = source.geometry

        volumes = {
            "albedo": generate_gridvolume(
                geometry,
                source.eval_albedo,
                units=ureg.dimensionless,
            ),
            "sigma_t": generate_gridvolume(
                geometry,
                source.eval_sigma_t,
                units=collision_coefficient_units,
            ),
        }

        medium = "piecewise" if self.is_piecewise else "eoheterogeneous"

        aabb_conf = {}
        if isinstance(geometry, SphericalShellGeometry):
            if any(r < 1 for r in self.extremum_resolution):
                raise ValueError(
                    "in spherical shell geometry, 'extremum_resolution' must "
                    "be >= 1 in all dimensions. got "
                    f"{self.extremum_resolution}."
                )
            extr_conf = {"type": "sphericalcoordsextremum"}
        elif isinstance(geometry, PlaneParallelGeometry):
            extr_conf = {"type": "gridextremum"}

            aabb_min = geometry.bbox.min.m_as(length_units)
            # The geometry bbox inserts a sub-surface padding. We want the data to be
            # evaluated on the entire geometry extent in x and y directions, but we
            # should avoid the wrap_mode affecting the z direction.
            aabb_min[2] = geometry.grid.levels[0].m_as(length_units)
            aabb_conf = {
                "aabb_min": aabb_min,
                "aabb_max": geometry.bbox.max.m_as(length_units),
            }

        result.update({"type": medium, **aabb_conf, **volumes})

        if self.extremum_resolution != (1, 1, 1):
            result["extremum"] = {
                **extr_conf,
                "resolution": self.extremum_resolution,
            }

        return result

    def _build_medium_params(self, source: EvalRadProps) -> dict[str, SceneParameter]:
        """
        Generate the parameter update map of a medium created with
        :meth:`_build_medium_template`.

        Parameters
        ----------
        source : .Atmosphere or .AtmosphereComponent
            Object whose radiative properties are evaluated.

        Returns
        -------
        dict
            Parameter update map.
        """
        collision_coefficient_units = uck.get("collision_coefficient")

        if self.is_homogeneous:
            # "value" appears twice because the mi.Spectrum is encapsulated in
            # a mi.ConstVolume
            return {
                key: SceneParameter(
                    lambda ctx, eval_fn=eval_fn, units=units: _uniform_value(
                        eval_fn(ctx.si), units
                    ),
                    KernelSceneParameterFlags.SPECTRAL,
                    search=SearchSceneParameter(
                        node_type=mi.Medium,
                        node_id=self.medium_id,
                        parameter_relpath=key,
                    ),
                )
                for key, eval_fn, units in [
                    ("albedo.value.value", source.eval_albedo, ureg.dimensionless),
                    (
                        "sigma_t.value.value",
                        source.eval_sigma_t,
                        collision_coefficient_units,
                    ),
                ]
            }

        geometry = source.geometry

        if isinstance(geometry, PlaneParallelGeometry):
            albedo_key = "albedo.data"
            sigma_t_key = "sigma_t.data"
        elif isinstance(geometry, SphericalShellGeometry):
            albedo_key = "albedo.volume.data"
            sigma_t_key = "sigma_t.volume.data"
        else:
            raise TypeError(
                f"unhandled scene geometry type '{type(geometry).__name__}'"
            )

        albedo_search = SearchSceneParameter(
            node_type=mi.Medium, node_id=self.medium_id, parameter_relpath=albedo_key
        )
        sigma_t_search = SearchSceneParameter(
            node_type=mi.Medium, node_id=self.medium_id, parameter_relpath=sigma_t_key
        )
        albedo_grid = generate_gridvolume(
            geometry,
            source.eval_albedo,
            units=ureg.dimensionless,
            search=albedo_search,
            flag=KernelSceneParameterFlags.SPECTRAL,
        )
        sigma_t_grid = generate_gridvolume(
            geometry,
            source.eval_sigma_t,
            units=collision_coefficient_units,
            search=sigma_t_search,
            flag=KernelSceneParameterFlags.SPECTRAL,
        )

        return {
            albedo_key: albedo_grid,
            sigma_t_key: sigma_t_grid,
        }

    @property
    def _template_phase(self) -> dict:
        """
        Phase function-related contribution to the kernel scene dictionary
        template.
        """
        return traverse(self.phase)[0].data

    @property
    def _template_medium(self) -> dict:
        """
        Medium-related contribution to the kernel scene dictionary template.
        """
        return self._build_medium_template(self)

    @property
    def _template_shape(self) -> dict:
        """
        Shape-related contribution to the kernel scene dictionary template.
        """
        return traverse(self.shape)[0].data

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
    def _params_phase(self) -> dict[str, SceneParameter]:
        """
        Phase function-related contribution to the parameter update map
        template.
        """
        umap = traverse(self.phase)[1].data

        # Add prefix and lookup strategy to all entries
        result = {}

        for uparam_key, uparam in umap.items():
            result[uparam_key] = attrs.evolve(
                uparam,
                search=SearchSceneParameter(
                    node_type=mi.PhaseFunction,
                    node_id=self.phase_id,
                    parameter_relpath=uparam_key,
                ),
            )

        return result

    @property
    def _params_medium(self) -> dict[str, SceneParameter]:
        """
        Medium-related contribution to the parameter update map template.
        """
        return self._build_medium_params(self)

    @property
    def _params_shape(self) -> dict[str, SceneParameter]:
        """
        Shape-related contribution to the parameter update map template.
        """
        return traverse(self.shape)[1].data

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
