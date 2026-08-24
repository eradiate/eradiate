from __future__ import annotations

import typing as t
from collections import abc as cabc

import attrs
import numpy as np

from ._core import PhaseFunction, phase_function_factory
from .._gridvolume import generate_gridvolume
from ..core import traverse
from ..geometry import PlaneParallelGeometry, SceneGeometry, SphericalShellGeometry
from ...attrs import documented
from ...contexts import KernelContext
from ...kernel import KernelSceneParameterFlags, SceneParameter
from ...spectral.index import SpectralIndex
from ...util.misc import cache_by_id


def _weights_converter(x):
    if isinstance(x, np.ndarray):
        return x
    dt_value = x
    for _ in range(np.ndim(dt_value)):
        dt_value = dt_value[0]
    if callable(dt_value):
        return x
    return np.array(x, dtype=np.float64)


@attrs.define(eq=False, slots=False)
class MultiPhaseFunction(PhaseFunction):
    """
    Multi phase function.

    Aggregates two or more sub-phase functions
    (*components*) and blends them based on its `weights` parameter.
    Weights are typically the associated medium's scattering coefficient.
    """

    components: list[PhaseFunction] = documented(
        attrs.field(
            converter=lambda x: [phase_function_factory.convert(y) for y in x],
            validator=attrs.validators.deep_iterable(
                attrs.validators.instance_of(PhaseFunction)
            ),
            kw_only=True,
        ),
        type="list of :class:`.PhaseFunction`",
        init_type="list of :class:`.PhaseFunction` or list of dict",
        doc="List of components (at least two). This parameter has no default.",
    )

    @components.validator
    def _components_validator(self, attribute, value):
        if not len(value) > 1:
            raise ValueError(
                f"while validating {attribute.name}: MultiPhaseFunction must "
                "have at least two components"
            )

    weights: np.ndarray | list[t.Callable[[KernelContext], np.ndarray]] = documented(
        attrs.field(
            converter=_weights_converter,
            kw_only=True,
        ),
        type="ndarray or list of callables",
        init_type="array-like or list of callables",
        doc="List of weights, one per component, in the same order as "
        "``components``. Weights may be numerical values: an array of shape "
        "``(n_components, [n_x, [n_y, ]]n_z)`` -- the leading component axis "
        "is required, the trailing spatial axes are optional and, when "
        "present, must be given innermost-last (``n_z`` alone, then "
        "``(n_x, n_z)``, then ``(n_x, n_y, n_z)``). Alternatively, weights "
        "may be a list of ``n_components`` callables, each taking a "
        ":class:`.KernelContext` as argument and returning *its own* "
        "component's weight array in one of those same spatial shapes, "
        "without the leading component axis -- the per-component arrays are "
        "stacked into the full tensor internally, not returned pre-stacked "
        "by a single callable. This parameter is required and has no "
        "default.",
    )

    @weights.validator
    def _weights_validator(self, attribute, value):
        self._weights_validator_impl(attribute, value)

    geometry: SceneGeometry | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(SceneGeometry.convert),
            validator=attrs.validators.optional(
                attrs.validators.instance_of(SceneGeometry)
            ),
        ),
        doc="Parameters defining the geometry of the scene. If unset, "
        "the volume textures defining component weights will be assigned "
        "default shape and extents likely unsuitable for atmosphere "
        "construction.",
        type=".SceneGeometry or None",
        init_type=".SceneGeometry or dict or str, optional",
        default="None",
    )

    def update(self) -> None:
        super().update()

        # Synchronize geometries
        for component in self.components:
            component.update()

            if isinstance(component, MultiPhaseFunction):
                component.geometry = self.geometry

    def _weights_validator_impl(self, attribute, value):
        if isinstance(value, np.ndarray):
            if value.ndim > 4:
                raise ValueError(
                    f"while validating '{attribute.name}': array must have at "
                    "most 4 dimensions (n_components, [n_x, [n_y, ]]n_z), got "
                    f"{value.ndim}"
                )
            if not value.shape[0] == len(self.components):
                raise ValueError(
                    f"while validating '{attribute.name}': array must have shape "
                    "(n, ...) where n is the number of components; got "
                    f"{value.shape}",
                )
        elif isinstance(value, cabc.Sequence):
            if not len(value) == len(self.components):
                raise ValueError(
                    f"while validating '{attribute.name}': weight and component "
                    f"lists must have the same length"
                )

    use_mis: bool = documented(
        attrs.field(
            converter=lambda x: bool(x),
            validator=attrs.validators.instance_of(bool),
            kw_only=True,
            default=True,
        ),
        type="bool",
        init_type="bool",
        doc="Use multiple importance sampling. Defaults to True.",
    )

    @cache_by_id
    def _eval_weights_impl(self, si: SpectralIndex) -> np.ndarray:
        if isinstance(self.weights, list):
            weights = []
            for weight_func in self.weights:
                weights.append(weight_func(si))
            weights = np.stack(weights, dtype=np.float32)

        else:  # if isinstance(self.weights, np.ndarray):
            weights = np.array(self.weights, dtype=np.float32)

        return weights

    def _eval_weights(
        self,
        si: SpectralIndex,
        n_component: int | list[int] | None = None,
    ) -> np.ndarray:
        weights = self._eval_weights_impl(si)

        if n_component is None:
            n_component = range(len(self.components))

        return weights[n_component, ...]

    @property
    def template(self):
        if not isinstance(
            self.geometry, (type(None), PlaneParallelGeometry, SphericalShellGeometry)
        ):
            raise TypeError(
                f"unhandled scene geometry type '{type(self.geometry).__name__}'"
            )

        result = {"type": "multiphase", "use_mis": self.use_mis}

        for i in range(len(self.components)):
            template, _ = traverse(self.components[i])
            result.update(
                {f"phase_{i}.{k}": v for k, v in template.items()},
            )

            result[f"weight_{i}"] = generate_gridvolume(
                self.geometry,
                self._eval_weights,
                extra_kwargs={"n_component": i},
            )

        return result

    @property
    def params(self) -> dict[str, SceneParameter]:
        result = {}

        if not isinstance(
            self.geometry, (type(None), PlaneParallelGeometry, SphericalShellGeometry)
        ):
            raise TypeError(
                f"unhandled scene geometry type '{type(self.geometry).__name__}'"
            )

        for i in range(len(self.components)):
            # Add components
            _, params = traverse(self.components[i])
            result.update({f"phase_{i}.{k}": v for k, v in params.items()})

            weight_param = generate_gridvolume(
                self.geometry,
                self._eval_weights,
                flag=KernelSceneParameterFlags.SPECTRAL,
                extra_kwargs={"n_component": i},
            )

            if self.geometry is None or isinstance(
                self.geometry, PlaneParallelGeometry
            ):
                result[f"weight_{i}.data"] = weight_param
            elif isinstance(self.geometry, SphericalShellGeometry):
                result[f"weight_{i}.volume.data"] = weight_param

        return result

    def with_normalized_weights(self, ctx: KernelContext) -> MultiPhaseFunction:
        """
        Return a copy of this phase function with per-cell weights rescaled
        to sum to 1 across components.

        The returned instance's ``weights`` is always a plain ``ndarray``,
        even if ``self.weights`` was a list of callables: this evaluates
        them (and the normalization) at ``ctx.si``, so the result is a
        snapshot at that one spectral index, not a phase function that
        stays normalized under further context changes.

        Parameters
        ----------
        ctx : .KernelContext
            Context at which weights are evaluated before normalization.

        Returns
        -------
        .MultiPhaseFunction
            A new instance with the same components and geometry, and
            weights normalized to sum to 1 in every cell.
        """
        weights = self._eval_weights_impl(ctx.si)

        # A cell where every component's weight is 0 only arises if sigma_t
        # is also 0 there, in which case the phase function is never
        # evaluated; this is physically unreachable outside contrived test
        # setups, so the resulting NaN is harmless.
        sum_weights = weights.sum(axis=0, keepdims=True)
        weights_normalized = weights / sum_weights

        return MultiPhaseFunction(
            components=self.components,
            weights=weights_normalized,
            geometry=self.geometry,
            use_mis=self.use_mis,
        )
