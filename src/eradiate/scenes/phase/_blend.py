import typing as t

import attrs
import mitsuba as mi
import numpy as np

from ._core import PhaseFunction, PhaseFunctionNode, phase_function_factory
from ..core import BoundingBox, Param, traverse
from ...attrs import documented, parse_docs
from ...kernel.transform import map_unit_cube
from ...units import unit_context_kernel as uck


def _weights_converter(value: np.typing.ArrayLike) -> np.ndarray:
    """
    Normalise weights so that their sum is 1.
    If all weights are zeros, leave their values at 0.
    """
    weights = np.array(value, dtype=np.float64)  # Ensure conversion of int to float
    weights_sum = weights.sum(axis=0)
    return np.divide(
        weights,
        weights_sum,
        where=weights_sum != 0.0,
        out=np.zeros_like(weights),
    )


@parse_docs
@attrs.define(eq=False, slots=False)
class BlendPhaseFunction(PhaseFunctionNode):
    """
    Blended phase function [``blend_phase``].

    This phase function aggregates two or more sub-phase functions
    (*components*) and blends them based on its `weights` parameter. Weights are
    usually based on the associated medium's scattering coefficient.
    """

    components: t.List[PhaseFunction] = documented(
        attrs.field(
            converter=lambda x: [phase_function_factory.convert(y) for y in x],
            validator=attrs.validators.deep_iterable(
                attrs.validators.instance_of(PhaseFunctionNode)
            ),
            kw_only=True,
        ),
        type="list of :class:`.PhaseFunction`",
        init_type="list of :class:`.PhaseFunction` or list of dict",
        doc="List of components (at least two). This parameter has not default.",
    )

    @components.validator
    def _components_validator(self, attribute, value):
        if not len(value) > 1:
            raise ValueError(
                f"while validating {attribute.name}: BlendPhaseFunction must "
                "have at least two components"
            )

    weights: np.ndarray = documented(
        attrs.field(
            converter=_weights_converter,
            kw_only=True,
        ),
        type="ndarray",
        init_type="array-like",
        doc="List of weights associated with each component. Must be of shape "
        "(n,) or (n, m) where n is the number of components and m the number "
        "of cells along the atmosphere's vertical axis. This parameter has no "
        "default.",
    )

    @weights.validator
    def _weights_validator(self, attribute, value):
        if value.ndim == 0 or value.ndim > 2:
            raise ValueError(
                f"while validating '{attribute.name}': array must have 1 or 2 "
                f"dimensions, got {value.ndim}"
            )

        if not value.shape[0] == len(self.components):
            raise ValueError(
                f"while validating '{attribute.name}': array must have shape "
                f"(n,) or (n, m) where n is the number of components; got "
                f"{value.shape}"
            )

    bbox: t.Optional[BoundingBox] = documented(
        attrs.field(
            default=None,
            converter=BoundingBox.convert,
            validator=attrs.validators.optional(
                attrs.validators.instance_of(BoundingBox)
            ),
        ),
        default="None",
        type=":class:`.BoundingBox` or None",
        init_type="quantity or array-like or :class:`.BoundingBox`, optional",
        doc="Optional bounding box describing the extent of the volume "
        "associated with this phase function.",
    )

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    def _gridvolume_transform(self) -> "mitsuba.ScalarTransform4f":
        if self.bbox is None:
            # This is currently possible because the bounding box is expected to
            # be set by a parent Atmosphere object based on the selected geometry
            raise ValueError(
                "computing the gridvolume transform requires a bounding box"
            )

        length_units = uck.get("length")
        bbox_min = self.bbox.min.m_as(length_units)
        bbox_max = self.bbox.max.m_as(length_units)

        return map_unit_cube(
            xmin=bbox_min[0],
            xmax=bbox_max[0],
            ymin=bbox_min[1],
            ymax=bbox_max[1],
            zmin=bbox_min[2],
            zmax=bbox_max[2],
        )

    @property
    def template(self) -> dict:
        result = {"type": "blendphase"}

        # The second kernel component aggregates all components except for the
        # first. Consequently, its weight is equal to the sum of all weights,
        # except for the first one.
        weight = self.weights[1:, ...].sum(axis=0)
        template_1, _ = traverse(self.components[0])

        if len(self.components) == 2:
            template_2, _ = traverse(self.components[1])

        else:
            # The phase function corresponding to remaining components
            # is built recursively
            template_2, _ = traverse(
                BlendPhaseFunction(
                    id=f"{self.id}_phase2",
                    components=self.components[1:],
                    weights=self.weights[1:, ...],
                    bbox=self.bbox,
                )
            )

        result.update(
            {
                **{f"phase1.{k}": v for k, v in template_1.items()},
                **{f"phase2.{k}": v for k, v in template_2.items()},
            }
        )

        # Pass weight values either as a GridVolume or a scalar
        if self.weights.ndim == 2 and self.weights.shape[1] > 1:
            # Mind dim ordering! (C-style, i.e. zyx)
            values = np.reshape(weight, (-1, 1, 1))
            grid_weight = mi.VolumeGrid(values.astype(np.float32))
            result.update({"weight.type": "gridvolume", "weight.grid": grid_weight})

            if self.bbox is not None:
                result["weight.to_world"] = self._gridvolume_transform()

        else:
            result["weight"] = float(weight)

        return result

    @property
    def params(self) -> t.Dict[str, Param]:
        result = {}

        _, params_1 = traverse(self.components[0])

        if len(self.components) == 2:
            _, params_2 = traverse(self.components[1])

        else:
            # The phase function corresponding to remaining components
            # is built recursively
            n_components = len(self.components) - 1
            _, params_2 = traverse(
                BlendPhaseFunction(
                    id=f"{self.id}_phase2",
                    components=self.components[1:],
                    weights=[1.0 / n_components for _ in range(n_components)],
                    bbox=None,
                )
            )

        result.update(
            {
                **{f"phase1.{k}": v for k, v in params_1.items()},
                **{f"phase2.{k}": v for k, v in params_2.items()},
            }
        )
        return result
