"""
Heterogeneous atmospheres.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import MutableMapping
from typing import TYPE_CHECKING

import attrs
import mitsuba as mi
import numpy as np
import pint

from ._core import AbstractHeterogeneousAtmosphere, atmosphere_factory
from ._molecular import MolecularAtmosphere
from ._particle_component import ParticleComponent
from ..core import traverse
from ..phase import MultiPhaseFunction, PhaseFunction
from ...attrs import define, documented
from ...grid import SpatialGrid
from ...kernel import SearchSceneParameter
from ...spectral.index import SpectralIndex
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg
from ...util.deprecation import DeprecatedWarning, renamed_kwargs
from ...util.misc import cache_by_id

if TYPE_CHECKING:
    from axsdb import AbsorptionDatabase

logger = logging.getLogger(__name__)


def _molecular_converter(value):
    if isinstance(value, MutableMapping) and ("type" not in value):
        value["type"] = "molecular"
    return atmosphere_factory.convert(value, allowed_cls=MolecularAtmosphere)


def _particles_converter(value):
    if not value:
        return []
    if not isinstance(value, (list, tuple)):
        return _particles_converter([value])
    return [ParticleComponent.convert(element) for element in value]


#: Component fields overridden by the parent :class:`.HeterogeneousAtmosphere`
_COMPONENT_IGNORED_FIELDS = (
    "use_rrt",
    "ddis_threshold",
    "force_majorant",
    "extremum_resolution",
)


def _warn_ignored_component_fields(
    attribute: attrs.Attribute,
    component: AbstractHeterogeneousAtmosphere,
) -> None:
    """
    Warn if a component sets a field that has no effect when it is nested in a
    :class:`.HeterogeneousAtmosphere`.
    """
    fields = attrs.fields_dict(type(component))
    for name in _COMPONENT_IGNORED_FIELDS:
        if not np.array_equal(getattr(component, name), fields[name].default):
            warnings.warn(
                f"while validating {attribute.name}: component field "
                f"'{name}' has no effect and is ignored; set "
                f"'HeterogeneousAtmosphere.{name}' instead",
                UserWarning,
                stacklevel=2,
            )


@renamed_kwargs(
    {"particle_layers": "particle_components"},
    deprecated_in="1.3.0",
    removed_in="2.0.0",
)
@define(eq=False, slots=False)
class HeterogeneousAtmosphere(AbstractHeterogeneousAtmosphere):
    """
    Heterogeneous atmosphere scene element [``heterogeneous``].

    Supports both 1D (layer-based) and 3D (fully gridded)
    configurations.
    """

    molecular_atmosphere: MolecularAtmosphere | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(_molecular_converter),
            validator=attrs.validators.optional(
                attrs.validators.instance_of(MolecularAtmosphere)
            ),
        ),
        doc="Molecular atmosphere. May be specified as a dictionary interpreted "
        'by :data:`.atmosphere_factory`; in that case, the ``"type"`` parameter '
        'may be omitted and will automatically be set to ``"molecular"``.',
        type=".MolecularAtmosphere or None",
        init_type=".MolecularAtmosphere or dict, optional",
        default="None",
    )

    @molecular_atmosphere.validator
    def _molecular_atmosphere_validator(self, attribute, value):
        if value is None:
            return

        if value.scale is not None:
            raise ValueError(
                f"while validating {attribute.name}: components cannot be "
                "scaled individually"
            )

        _warn_ignored_component_fields(attribute, value)

    particle_components: list[ParticleComponent] = documented(
        attrs.field(
            factory=list,
            converter=_particles_converter,
            validator=attrs.validators.deep_iterable(
                attrs.validators.instance_of(ParticleComponent)
            ),
        ),
        doc="List of particle components. Elements may be specified as "
        "dictionaries interpreted by :data:`.atmosphere_factory`; in that "
        "case, the ``type`` parameter may be omitted and will automatically "
        'be set to ``"particle_ot"``.',
        type="list of .ParticleComponent",
        init_type="list of .ParticleComponent, optional",
        default="[]",
    )

    @particle_components.validator
    def _particles_validator(self, attribute, value):
        if not all(component.scale is None for component in value):
            raise ValueError(
                f"while validating {attribute.name}: components cannot be "
                "scaled individually"
            )

        for component in value:
            _warn_ignored_component_fields(attribute, component)

    use_phase_mis: bool = documented(
        attrs.field(default=True, converter=bool, kw_only=True),
        doc="If ``True``, multiple importance sampling is enabled for the "
        "3D phase function mixture.",
        type="bool",
        init_type="bool",
        default="True",
    )

    @property
    def particle_layers(self) -> list[ParticleComponent]:
        """
        .. deprecated:: 1.3.0
           Use :attr:`particle_components` instead.
        """
        warnings.warn(
            DeprecatedWarning(
                component="'HeterogeneousAtmosphere.particle_layers'",
                deprecated_in="1.3.0",
                removed_in="2.0.0",
                details="Use 'particle_components' instead.",
            ),
            stacklevel=2,
        )
        return self.particle_components

    @property
    def components(self) -> list[MolecularAtmosphere | ParticleComponent]:
        """
        Returns
        -------
        list of .MolecularAtmosphere or .ParticleComponent
            The list of all registered atmospheric components.
        """
        result = [self.molecular_atmosphere] if self.molecular_atmosphere else []
        result.extend(self.particle_components)
        return result

    def update(self):
        # Inherit docstring
        if not self.components:
            raise ValueError("HeterogeneousAtmosphere must have at least one component")

        # Force component IDs
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
    #              Spatial extension and thermophysical properties
    # --------------------------------------------------------------------------

    @property
    def bottom(self) -> pint.Quantity:
        # Inherit docstring
        return min([component.bottom for component in self.components])

    @property
    def top(self) -> pint.Quantity:
        # Inherit docstring
        return max([component.top for component in self.components])

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    @property
    def absorption_data(self) -> AbsorptionDatabase | None:
        # Inherit docstring
        if self.molecular_atmosphere is not None:
            return self.molecular_atmosphere._absorption_data
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
        # Inherit docstring
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
    def _template_phase(self) -> dict:
        # Inherit docstring
        return traverse(self.phase)[0].data

    @property
    def _params_phase(self) -> dict:
        # Inherit docstring
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
