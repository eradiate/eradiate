from __future__ import annotations

import typing as t
from collections import abc as cabc

import attrs
import mitsuba as mi
import numpy as np

from ._core import Abstract1DBlendPhaseFunction, phase_function_factory
from ..core import traverse
from ..geometry import PlaneParallelGeometry, SceneGeometry, SphericalShellGeometry
from ...attrs import documented
from ...contexts import KernelContext
from ...kernel import DictParameter, KernelSceneParameterFlags, SceneParameter
from ...spectral.index import SpectralIndex
from ...util.misc import cache_by_id


@attrs.define(eq=False, slots=False)
class Multi1DPhaseFunction(Abstract1DBlendPhaseFunction):


    @cache_by_id
    def _eval_weights(
        self,
        si: SpectralIndex,
        n_component: int | list[int] | None = None,
    ) -> np.array:

        if isinstance(self.weights, list):
            weights = np.array([w(si) for w in self.weights], dtype=np.float64)
        else:  # if isinstance(self.weights, np.ndarray):
            weights = np.array(self.weights, dtype=np.float64)

        if weights.ndim < 2:
            weights = weights.reshape((-1, 1))
        
        if n_component is None:
            n_component = range(len(self.components) - 1)
        elif isinstance(n_component, int):
            n_component = [n_component]

        return weights[n_component, ...]

    
    @property
    def template(self):

        result = {"type": "multiphase"}

        for i in range(len(self.components)):

            if self.geometry is None or isinstance(self.geometry, PlaneParallelGeometry):
                dim_order=(-1, 1, 1)
            elif isinstance(self.geometry, SphericalShellGeometry):
                dim_order=(1, 1, -1)
            else:
                raise ValueError(
                    f"unhandled scene geometry type '{type(self.geometry).__name__}'"
                )

            template, _ = traverse(self.components[i])
            result.update(
                {
                    **{f"phase_1.phase{i}.{k}": v for k, v in template.items()},
                    #f"{prefix}.type": "blendphase",
                }
            )

            # Note: This defines a partial and evaluates the component index.
            # Passing i as the kwarg default value is essential to force the
            # dereferencing of the loop variable.
            def eval_weights(ctx: KernelContext, n_component=i):
                return mi.VolumeGrid(
                    np.reshape(
                        self.eval_weights(ctx.si, n_component),
                        dim_order,  # Mind dim ordering! (C-style, i.e. zyx)
                    ).astype(np.float32)
                )

            if self.geometry is None or isinstance(
                self.geometry, PlaneParallelGeometry
            ):

                result[f"phase_1.weight{i}.type"] = "gridvolume"
                result[f"phase_1.weight{i}.grid"] = DictParameter(eval_weights)
                result[f"phase_1.weight{i}.filter_type"] = "nearest"

                if self.geometry is not None:
                    result[f"phase_1.weight.to_world"] = (
                        self.geometry.atmosphere_volume_to_world
                    )

            elif isinstance(self.geometry, SphericalShellGeometry):
                
                result[f"phase_1.weight{i}.type"] = "sphericalcoordsvolume"
                result[f"phase_1.weight{i}.volume.type"] = "gridvolume"
                result[f"phase_1.weight{i}.volume.grid"] = DictParameter(
                    eval_weights
                )
                result[f"phase_1.weight{i}.volume.filter_type"] = "nearest"
                result[f"phase_1.weight{i}.to_world"] = (
                    self.geometry.atmosphere_volume_to_world
                )
                result[f"phase_1.weight{i}.rmin"] = self.geometry.atmosphere_volume_rmin


    @property
    def params(self) -> dict[str, SceneParameter]:
        

        result = {}

        for i in range(len(self.components) - 1):

            # Add components
            _, params = traverse(self.components[i])
            result.update(
                {
                    **{f"phase_1.phase{i}.{k}": v for k, v in params.items()},
                }
            )

            if self.geometry is None or isinstance(
                self.geometry, PlaneParallelGeometry
            ):
                # Note: This defines a partial and evaluates the component index.
                # Passing i as the kwarg default value is essential to force the
                # dereferencing of the loop variable.
                def eval_conditional_weights(ctx: KernelContext, n_component=i):
                    return np.reshape(
                        self.eval_conditional_weights(ctx.si, n_component),
                        (-1, 1, 1, 1),  # Mind dim ordering! (C-style, i.e. zyxc)
                    ).astype(np.float32)

                # Assign conditional weight to second component
                result[f"phase_1.weight{i}.data"] = SceneParameter(
                    eval_conditional_weights,
                    KernelSceneParameterFlags.SPECTRAL,
                )

            elif isinstance(self.geometry, SphericalShellGeometry):
                # Same comment as above
                def eval_conditional_weights(ctx: KernelContext, n_component=i):
                    return np.reshape(
                        self.eval_conditional_weights(ctx.si, n_component),
                        (1, 1, -1, 1),  # Mind dim ordering! (C-style, i.e. zyxc)
                    ).astype(np.float32)

                # Assign conditional weight to second component
                result[f"phase_1.weight{i}.volume.data"] = SceneParameter(
                    eval_conditional_weights,
                    KernelSceneParameterFlags.SPECTRAL,
                )

            else:
                raise NotImplementedError
        
        return result