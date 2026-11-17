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
import pint

from ._core import (
    AbstractHeterogeneousAtmosphere,
    AtmosphereComponent,
    atmosphere_factory,
)
from ._merge import (
    MergeAll,
    MergedComponent,
    MergePolicy,
    _same_geometry,
    merge_policy_converter,
)
from ._molecular import MolecularComponent
from ._particle_component import ParticleComponent
from ..core import traverse
from ..phase import PhaseFunction
from ...attrs import define, documented
from ...grid import SpatialGrid
from ...kernel import SearchSceneParameter
from ...spectral.index import SpectralIndex
from ...util.deprecation import DeprecatedWarning, renamed_kwargs
from ...util.misc import flatten

if TYPE_CHECKING:
    from axsdb import AbsorptionDatabase

logger = logging.getLogger(__name__)


def _molecular_converter(value):
    if isinstance(value, MutableMapping) and ("type" not in value):
        value["type"] = "molecular"
    return atmosphere_factory.convert(value, allowed_cls=MolecularComponent)


def _particles_converter(value):
    if not value:
        return []
    if not isinstance(value, (list, tuple)):
        return _particles_converter([value])
    return [ParticleComponent.convert(element) for element in value]


def _components_converter(value):
    if not value:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [
        atmosphere_factory.convert(element, allowed_cls=AtmosphereComponent)
        for element in value
    ]


def _flatten_components(
    components: list[AtmosphereComponent],
) -> list[AtmosphereComponent]:
    return [
        member
        for component in components
        for member in (
            component.components
            if isinstance(component, MergedComponent)
            else [component]
        )
    ]


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

    components: list[AtmosphereComponent] = documented(
        attrs.field(
            factory=list,
            converter=_components_converter,
            validator=attrs.validators.deep_iterable(
                attrs.validators.instance_of(AtmosphereComponent)
            ),
        ),
        doc="List of atmosphere components, *i.e.* a :class:`.MolecularComponent` "
        ":class:`.ParticleComponent`. Elements may be specified as dictionaries "
        "interpreted by :data:`.atmosphere_factory`; in that case, the "
        "``type`` parameter is required. At most one "
        ":class:`.MolecularComponent` is allowed. :class:`.MergedComponent` "
        "elements are kept as they are; ``merge_policy`` applies to the other "
        "components.",
        type="list of .AtmosphereComponent",
        init_type="list of .AtmosphereComponent or dict, optional",
        default="[]",
    )

    @components.validator
    def _components_validator(self, attribute, value):
        members = _flatten_components(value)

        if sum(isinstance(c, MolecularComponent) for c in members) > 1:
            raise ValueError(
                f"while validating {attribute.name}: at most one "
                "MolecularComponent is allowed"
            )

        if len({id(c) for c in members}) != len(members):
            raise ValueError(
                f"while validating {attribute.name}: a component may appear only once"
            )

        if not all(component.scale is None for component in [*value, *members]):
            raise ValueError(
                f"while validating {attribute.name}: components cannot be "
                "scaled individually"
            )

    merge_policy: MergePolicy = documented(
        attrs.field(
            factory=MergeAll,
            converter=merge_policy_converter,
            validator=attrs.validators.instance_of(MergePolicy),
            kw_only=True,
        ),
        doc="Policy merging the components which are not already part of a "
        ":class:`.MergedComponent`. Each merged component is rendered as one "
        "medium; several merged components are rendered as overlapping media. "
        "May be specified as a dictionary interpreted by "
        ":data:`.merge_policy_factory` or as a registered type ID, *e.g.* "
        '``"merge_all"``, ``"overlap_all"``, ``"merge_by_grid"``.',
        type=".MergePolicy",
        init_type=".MergePolicy or dict or str, optional",
        default=":class:`MergeAll() <.MergeAll>`",
    )

    use_phase_mis: bool = documented(
        attrs.field(default=True, converter=bool, kw_only=True),
        doc="If ``True``, multiple importance sampling is enabled for the "
        "3D phase function mixture.",
        type="bool",
        init_type="bool",
        default="True",
    )

    components_merged: list[MergedComponent] = documented(
        attrs.field(factory=list, init=False, repr=False),
        doc="Merged components generated from ``components`` by "
        ":meth:`merge_components`.",
        type="list of .MergedComponent",
    )

    def __init__(
        self, *args, molecular_atmosphere=None, particle_components=None, **kwargs
    ):
        if molecular_atmosphere is not None or particle_components is not None:
            if "components" in kwargs:
                raise TypeError(
                    "cannot pass 'components' together with "
                    "'molecular_atmosphere' or 'particle_components'"
                )

            for name, value in [
                ("molecular_atmosphere", molecular_atmosphere),
                ("particle_components", particle_components),
            ]:
                if value is not None:
                    warnings.warn(
                        DeprecatedWarning(
                            component=f"'HeterogeneousAtmosphere.{name}'",
                            deprecated_in="1.3.1",
                            removed_in="2.0.0",
                            details="Use 'components' instead.",
                        ),
                        stacklevel=3,
                    )

            # Molecular first in order to match legacy input
            components = []
            if molecular_atmosphere is not None:
                components.append(_molecular_converter(molecular_atmosphere))
            components += _particles_converter(particle_components)

            kwargs["components"] = components

        self.__attrs_init__(*args, **kwargs)

    @property
    def molecular_atmosphere(self) -> MolecularComponent | None:
        """
        Molecular component of this atmosphere, if any.
        """
        return next(
            (
                c
                for c in _flatten_components(self.components)
                if isinstance(c, MolecularComponent)
            ),
            None,
        )

    @property
    def particle_components(self) -> list[AtmosphereComponent]:
        """
        Particle components of this atmosphere, if any.
        """
        return [
            c
            for c in _flatten_components(self.components)
            if not isinstance(c, MolecularComponent)
        ]

    @property
    def is_overlapping(self) -> bool:
        """
        ``True`` if this atmosphere is rendered as overlapping media, *i.e.*
        if it has more than one merged component.
        """
        return len(self.components_merged) > 1

    @property
    def is_piecewise(self) -> bool:
        """
        ``True`` if this atmosphere is rendered with a single ``piecewise``
        medium (see :attr:`.AbstractHeterogeneousAtmosphere.is_piecewise`),
        evaluated on the geometry of its merged component. Overlapping media
        are never piecewise.
        """
        return not self.is_overlapping and self._is_piecewise_on(
            self.components_merged[0].geometry
        )

    @property
    def particle_layers(self) -> list[ParticleComponent]:
        """
        Particle components of this atmosphere.

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

    def update(self):
        # Inherit docstring
        if not self.components:
            raise ValueError("HeterogeneousAtmosphere must have at least one component")

        members = _flatten_components(self.components)

        for i, component in enumerate(members):
            component.update()
            component.id = f"{self.id}_component_{i}"
            if component._own_geometry is None:
                logger.debug(
                    "Override the geometry of component '%s' to %s (was %s)",
                    component.id,
                    self.geometry,
                    component.geometry,
                )
                component.geometry = self.geometry
            else:
                component.geometry = component._own_geometry

        self._merged_all = MergedComponent(
            id=self.id,
            components=members,
            geometry=self.geometry,
            use_phase_mis=self.use_phase_mis,
        )
        self.merge_components()

    def merge_components(self) -> None:
        """
        Populate :attr:`components_merged`: components which are not part of a
        :class:`.MergedComponent` are merged according to
        :attr:`merge_policy`, then merged components from :attr:`components`
        are appended. Merged components without a geometry get the geometry of
        this atmosphere, and their members are evaluated on it.

        This method is called by :meth:`update`.
        """
        loose = [c for c in self.components if not isinstance(c, MergedComponent)]
        groups = self.merge_policy.merge(loose, self.geometry) if loose else []
        groups += [c for c in self.components if isinstance(c, MergedComponent)]

        self.components_merged = []

        for i, group in enumerate(groups):
            geometry = (
                self.geometry if group._own_geometry is None else group._own_geometry
            )
            if type(geometry) is not type(self.geometry):
                raise ValueError(
                    f"merged component geometry type '{type(geometry).__name__}' "
                    f"differs from the atmosphere geometry type "
                    f"'{type(self.geometry).__name__}'"
                )

            for member in group.components:
                if member._own_geometry is not None and not _same_geometry(
                    member._own_geometry, geometry
                ):
                    warnings.warn(
                        f"component '{member.id}' is evaluated on the geometry "
                        "of its merged component instead of its own",
                        UserWarning,
                        stacklevel=2,
                    )
                member.geometry = geometry

            self.components_merged.append(
                attrs.evolve(
                    group,
                    id=f"{self.id}_group_{i}",
                    geometry=geometry,
                    use_phase_mis=self.use_phase_mis,
                )
            )

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    @property
    def absorption_data(self) -> AbsorptionDatabase | None:
        # Inherit docstring
        if self.molecular_atmosphere is not None:
            return self.molecular_atmosphere.absorption_data
        else:
            return None

    def eval_albedo(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        return self._merged_all.eval_albedo(si, grid)

    def eval_sigma_t(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        return self._merged_all.eval_sigma_t(si, grid)

    def eval_sigma_a(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        return self._merged_all.eval_sigma_a(si, grid)

    def eval_sigma_s(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        return self._merged_all.eval_sigma_s(si, grid)

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def phase(self) -> PhaseFunction:
        # Inherit docstring
        return self._merged_all.phase

    @property
    def template(self) -> dict:
        # Inherit docstring
        if not self.is_overlapping:
            return super().template

        phases = {
            group.phase_id: traverse(group.phase)[0].data
            for group in self.components_merged
        }
        result = flatten(
            {
                **phases,
                self.medium_id: self._template_medium,
                self.shape_id: self._template_shape,
            }
        )
        result.update(
            {
                f"{self.shape_id}.bsdf.type": "null",
                f"{self.shape_id}.interior.type": "ref",
                f"{self.shape_id}.interior.id": self.medium_id,
            }
        )

        return result

    @property
    def _template_phase(self) -> dict:
        # Inherit docstring
        return traverse(self.components_merged[0].phase)[0].data

    @property
    def _template_medium(self) -> dict:
        # Inherit docstring
        if not self.is_overlapping:
            return self._build_medium_template(
                self.components_merged[0], piecewise=self.is_piecewise
            )

        result = {"type": "overlap", "ddis_threshold": self.ddis_threshold}

        for i, group in enumerate(self.components_merged):
            result[f"component{i}"] = {
                **self._build_medium_template(group, piecewise=False),
                "id": group.medium_id,
                "phase": {"type": "ref", "id": group.phase_id},
            }

        return result

    @property
    def _params_phase(self) -> dict:
        # Inherit docstring
        result = {}

        for group in self.components_merged:
            if self.is_overlapping:
                phase_id, prefix = group.phase_id, f"{group.phase_id}."
            else:
                phase_id, prefix = self.phase_id, ""

            for uparam_key, uparam in traverse(group.phase)[1].data.items():
                result[f"{prefix}{uparam_key}"] = attrs.evolve(
                    uparam,
                    search=SearchSceneParameter(
                        node_type=mi.PhaseFunction,
                        node_id=phase_id,
                        parameter_relpath=uparam_key,
                    ),
                )

        return result

    @property
    def _params_medium(self) -> dict:
        # Inherit docstring
        if not self.is_overlapping:
            return self._build_medium_params(self.components_merged[0], self.medium_id)

        result = {}

        for group in self.components_merged:
            params = self._build_medium_params(group, group.medium_id)
            result.update({f"{group.medium_id}.{k}": v for k, v in params.items()})

        return result
