import typing as t

import attrs
import mitsuba as mi
import numpy as np

from ._core import PhaseFunction, phase_function_factory
from ..core import BoundingBox, KernelDict
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext
from ...kernel.transform import map_unit_cube
from ...units import unit_context_kernel as uck
from ...util.misc import onedict_value


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
@attrs.define
class BlendPhaseFunction(PhaseFunction):
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
                attrs.validators.instance_of(PhaseFunction)
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
            raise ValueError(
                "computing the gridvolume transform requires a bounding box"
            )

        length_units = uck.get("length")
        width = self.bbox.extents[1].m_as(length_units)
        top = self.bbox.max[2].m_as(length_units)
        bottom = self.bbox.min[2].m_as(length_units)

        return map_unit_cube(
            xmin=-0.5 * width,
            xmax=0.5 * width,
            ymin=-0.5 * width,
            ymax=0.5 * width,
            zmin=bottom,
            zmax=top,
        )

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        # Build kernel dict recursively

        # Summed weights of all components except the first, used to
        # set up the kernel dictionary of the phase function
        weight = self.weights[1:, ...].sum(axis=0)
        phase_dict_1 = onedict_value(self.components[0].kernel_dict(ctx))

        if len(self.components) == 2:
            phase_dict_2 = onedict_value(self.components[1].kernel_dict(ctx))

        else:
            # The phase function corresponding to remaining components
            # is built recursively
            phase_dict_2 = onedict_value(
                BlendPhaseFunction(
                    id=f"{self.id}_phase2",
                    components=self.components[1:],
                    weights=self.weights[1:, ...],
                    bbox=self.bbox,
                )
                .kernel_dict(ctx)
                .data
            )

        # Pass weight values either as a GridVolume or a scalar
        if self.weights.ndim == 2 and self.weights.shape[1] > 1:
            # Mind dim ordering! (C-style, i.e. zyx)
            values = np.reshape(weight, (-1, 1, 1))
            grid_weight = mi.VolumeGrid(values.astype(np.float32))
            weight = {"type": "gridvolume", "grid": grid_weight}

            if self.bbox is not None:
                weight["to_world"] = self._gridvolume_transform()

        else:
            weight = float(weight)

        return KernelDict(
            {
                self.id: {
                    "type": "blendphase",
                    "phase1": phase_dict_1,
                    "phase2": phase_dict_2,
                    "weight": weight,
                }
            }
        )
