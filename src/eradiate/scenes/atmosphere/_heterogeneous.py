"""
Heterogeneous atmospheres.
"""
from __future__ import annotations

from collections import abc as cabc

import attrs
import mitsuba as mi
import numpy as np
import pint

from ._core import AbstractHeterogeneousAtmosphere, atmosphere_factory
from ._molecular_atmosphere import MolecularAtmosphere
from ._particle_layer import ParticleLayer
from ..core import traverse
from ..phase import BlendPhaseFunction, PhaseFunction
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext, SpectralContext
from ...kernel import TypeIdLookupStrategy
from ...radprops import ZGrid
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg
from ...util.misc import cache_by_id


def _molecular_converter(value):
    if isinstance(value, cabc.MutableMapping) and ("type" not in value):
        value["type"] = "molecular"
    return atmosphere_factory.convert(value, allowed_cls=MolecularAtmosphere)


def _particle_layer_converter(value):
    if not value:
        return []

    if not isinstance(value, (list, tuple)):
        return _particle_layer_converter([value])

    else:
        result = []

        for element in value:
            if isinstance(element, cabc.MutableMapping) and ("type" not in element):
                element["type"] = "particle_layer"
            result.append(
                atmosphere_factory.convert(element, allowed_cls=ParticleLayer)
            )

        return result


@parse_docs
@attrs.define(eq=False, slots=False)
class HeterogeneousAtmosphere(AbstractHeterogeneousAtmosphere):
    """
    Heterogeneous atmosphere scene element [``heterogeneous``].
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

    particle_layers: list[ParticleLayer] = documented(
        attrs.field(
            factory=list,
            converter=_particle_layer_converter,
            validator=attrs.validators.deep_iterable(
                attrs.validators.instance_of(ParticleLayer)
            ),
        ),
        doc="List of particle layers. Elements may be specified as "
        "dictionaries interpreted by :data:`.atmosphere_factory`; in that "
        "case, the ``type`` parameter may be omitted and will automatically "
        'be set to ``"particle_layer"``.',
        type="list of .ParticleLayer",
        init_type="list of .ParticleLayer, optional",
        default="[]",
    )

    @particle_layers.validator
    def _particle_layers_validator(self, attribute, value):
        if not all(component.scale is None for component in value):
            raise ValueError(
                f"while validating {attribute.name}: components cannot be "
                "scaled individually"
            )

    @property
    def components(self) -> list[MolecularAtmosphere | ParticleLayer]:
        """
        Returns
        -------
        list of .AbstractHeterogeneousAtmosphere
            The list of all registered atmospheric components.
        """
        result = [self.molecular_atmosphere] if self.molecular_atmosphere else []
        result.extend(self.particle_layers)
        return result

    def update(self):
        # Inherit docstring

        if not self.components:
            raise ValueError("HeterogeneousAtmosphere must have at least one component")

        # Force component IDs and geometry
        for i, component in enumerate(self.components):
            component.update()
            component.id = f"{self.id}_component_{i}"
            component.geometry = self.geometry

    # --------------------------------------------------------------------------
    #              Spatial extension and thermophysical properties
    # --------------------------------------------------------------------------

    def eval_mfp(self, ctx: KernelDictContext) -> pint.Quantity:
        # Inherit docstring
        mfp = [component.eval_mfp(ctx=ctx) for component in self.components]
        return max(mfp)

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    def eval_albedo(
        self, sctx: SpectralContext, zgrid: ZGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        if zgrid is not None and zgrid is not self.geometry.zgrid:
            raise ValueError("zgrid must be left unset or set to self.geometry.zgrid")

        units = ucc.get("collision_coefficient")
        sigma_s = self.eval_sigma_s(sctx).m_as(units)
        sigma_t = self.eval_sigma_t(sctx).m_as(units)
        albedo = np.zeros_like(sigma_s)
        np.divide(sigma_s, sigma_t, where=sigma_t != 0.0, out=albedo)

        return albedo * ureg.dimensionless

    @cache_by_id
    def _eval_sigma_t_impl(self, sctx: SpectralContext) -> pint.Quantity:
        result = np.zeros((len(self.components), len(self.geometry.zgrid.layers)))
        sigma_units = ucc.get("collision_coefficient")

        # Evaluate extinction for current component
        for i, component in enumerate(self.components):
            result[i] = component.eval_sigma_t(sctx, self.geometry.zgrid).m_as(
                sigma_units
            )

        return result * sigma_units

    def eval_sigma_t(
        self, sctx: SpectralContext, zgrid: ZGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        if zgrid is not None and zgrid is not self.geometry.zgrid:
            raise ValueError("zgrid must be left unset or set to self.geometry.zgrid")
        return self._eval_sigma_t_impl(sctx).sum(axis=0)

    def eval_sigma_a(
        self, sctx: SpectralContext, zgrid: ZGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        if zgrid is not None and zgrid is not self.geometry.zgrid:
            raise ValueError("zgrid must be left unset or set to self.geometry.zgrid")
        return self.eval_sigma_t(sctx) - self.eval_sigma_s(sctx)

    @cache_by_id
    def _eval_sigma_s_impl(self, sctx: SpectralContext) -> pint.Quantity:
        result = np.zeros((len(self.components), len(self.geometry.zgrid.layers)))
        sigma_units = ucc.get("collision_coefficient")

        # Evaluate scattering coefficient for current component
        for i, component in enumerate(self.components):
            result[i] = component.eval_sigma_s(sctx, self.geometry.zgrid).m_as(
                sigma_units
            )

        return result * sigma_units

    def _eval_sigma_s_component(
        self, sctx: SpectralContext, n_component: int
    ) -> pint.Quantity:
        return self._eval_sigma_s_impl(sctx)[n_component]

    def eval_sigma_s(
        self, sctx: SpectralContext, zgrid: ZGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        if zgrid is not None and zgrid is not self.geometry.zgrid:
            raise ValueError("zgrid must be left unset or set to self.geometry.zgrid")
        return self._eval_sigma_s_impl(sctx).sum(axis=0)

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def phase(self) -> PhaseFunction:
        # Inherit docstring
        if len(self.components) == 1:
            return self.components[0].phase

        else:
            components, weights = [], []
            sigma_units = ucc.get("collision_coefficient")

            for i, component in enumerate(self.components):
                components.append(component.phase)

                def eval_sigma_s(
                    sctx: SpectralContext, n_component: int = i
                ) -> np.ndarray:
                    return self._eval_sigma_s_component(sctx, n_component).m_as(
                        sigma_units
                    )

                weights.append(eval_sigma_s)

            return BlendPhaseFunction(
                components=components, weights=weights, geometry=self.geometry
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
            result[f"phase_function.{uparam_key}"] = attrs.evolve(
                uparam,
                lookup_strategy=TypeIdLookupStrategy(
                    node_type=mi.Medium,
                    node_id=self.medium_id,
                    parameter_relpath=f"phase_function.{uparam_key}",
                ),
            )

        return result
