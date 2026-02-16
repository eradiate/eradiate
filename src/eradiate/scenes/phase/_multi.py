from __future__ import annotations

import attrs
import mitsuba as mi
import numpy as np

from ._blend import Abstract1DBlendPhaseFunction, Abstract3DBlendPhaseFunction
from ..core import traverse
from ..geometry import PlaneParallelGeometry, SphericalShellGeometry
from ...attrs import documented
from ...contexts import KernelContext
from ...kernel import DictParameter, KernelSceneParameterFlags, SceneParameter
from ...spectral.index import SpectralIndex
from ...util.misc import cache_by_id


@attrs.define(eq=False, slots=False)
class Multi1DPhaseFunction(Abstract1DBlendPhaseFunction):
    use_mis: bool = documented(
        attrs.field(
            converter=lambda x: bool(x),
            validator=attrs.validators.instance_of(bool),
            kw_only=True,
            default=True,
        ),
        type="bool",
        init_type="bool",
        doc="Use multiple importance sampling. Default to True.",
    )

    @cache_by_id
    def _eval_weights_impl(self, si: SpectralIndex):
        if isinstance(self.weights, list):
            weights = np.array([w(si) for w in self.weights], dtype=np.float64)
        else:  # if isinstance(self.weights, np.ndarray):
            weights = np.array(self.weights, dtype=np.float64)

        if weights.ndim < 2:
            weights = weights.reshape((-1, 1))

        return weights

    def _eval_weights(
        self,
        si: SpectralIndex,
        n_component: int | list[int] | None = None,
    ) -> np.ndarray:
        weights = self._eval_weights_impl(si)

        if n_component is None:
            n_component = range(len(self.components) - 1)
        elif isinstance(n_component, int):
            n_component = [n_component]

        return weights[n_component, ...]

    @property
    def template(self):
        if self.geometry is None or isinstance(self.geometry, PlaneParallelGeometry):
            dim_order = (-1, 1, 1)
        elif isinstance(self.geometry, SphericalShellGeometry):
            dim_order = (1, 1, -1)
        else:
            raise ValueError(
                f"unhandled scene geometry type '{type(self.geometry).__name__}'"
            )

        result = {"type": "multiphase", "use_mis": self.use_mis}

        for i in range(len(self.components)):
            template, _ = traverse(self.components[i])
            result.update(
                {f"phase_{i}.{k}": v for k, v in template.items()},
            )

            # Note: This defines a partial and evaluates the component index.
            # Passing i as the kwarg default value is essential to force the
            # dereferencing of the loop variable.
            def eval_weights(ctx: KernelContext, n_component=i):
                return mi.VolumeGrid(
                    np.reshape(
                        self._eval_weights(ctx.si, n_component),
                        dim_order,
                    ).astype(np.float32)
                )

            if self.geometry is None or isinstance(
                self.geometry, PlaneParallelGeometry
            ):
                result[f"weight_{i}.type"] = "gridvolume"
                result[f"weight_{i}.grid"] = DictParameter(eval_weights)
                result[f"weight_{i}.filter_type"] = "nearest"

                if self.geometry is not None:
                    result[f"weight_{i}.to_world"] = (
                        self.geometry.atmosphere_volume_to_world
                    )

            elif isinstance(self.geometry, SphericalShellGeometry):
                result[f"weight_{i}.type"] = "sphericalcoordsvolume"
                result[f"weight_{i}.volume.type"] = "gridvolume"
                result[f"weight_{i}.volume.grid"] = DictParameter(eval_weights)
                result[f"weight_{i}.volume.filter_type"] = "nearest"
                result[f"weight_{i}.to_world"] = (
                    self.geometry.atmosphere_volume_to_world
                )
                result[f"weight_{i}.rmin"] = self.geometry.atmosphere_volume_rmin

        return result

    @property
    def params(self) -> dict[str, SceneParameter]:
        result = {}

        for i in range(len(self.components)):
            # Add components
            _, params = traverse(self.components[i])
            result.update({f"phase_{i}.{k}": v for k, v in params.items()})

            if self.geometry is None or isinstance(
                self.geometry, PlaneParallelGeometry
            ):
                # Note: This defines a partial and evaluates the component index.
                # Passing i as the kwarg default value is essential to force the
                # dereferencing of the loop variable.
                def eval_weights(ctx: KernelContext, n_component=i):
                    return np.reshape(
                        self._eval_weights(ctx.si, n_component),
                        (-1, 1, 1, 1),  # Mind dim ordering! (C-style, i.e. zyxc)
                    ).astype(np.float32)

                # Assign conditional weight to second component
                result[f"weight_{i}.data"] = SceneParameter(
                    eval_weights,
                    KernelSceneParameterFlags.SPECTRAL,
                )

            elif isinstance(self.geometry, SphericalShellGeometry):
                # Same comment as above
                def eval_weights(ctx: KernelContext, n_component=i):
                    return np.reshape(
                        self._eval_weights(ctx.si, n_component),
                        (1, 1, -1, 1),  # Mind dim ordering! (C-style, i.e. zyxc)
                    ).astype(np.float32)

                # Assign conditional weight to second component
                result[f"weight_{i}.volume.data"] = SceneParameter(
                    eval_weights,
                    KernelSceneParameterFlags.SPECTRAL,
                )

            else:
                raise NotImplementedError

        return result

    def normalized(self, ctx: KernelContext) -> Multi1DPhaseFunction:
        weights = self.weights
        if callable(self.weights[0]):
            weights = np.asarray([c(ctx.si) for c in self.weights])

        sum_weights = weights.sum(axis=0, keepdims=True)
        weights_normalized = weights / sum_weights

        return Multi1DPhaseFunction(
            components=self.components,
            weights=weights_normalized,
            geometry=self.geometry,
        )


@attrs.define(eq=False, slots=False)
class Multi3DPhaseFunction(Abstract3DBlendPhaseFunction):
    use_mis: bool = documented(
        attrs.field(
            converter=lambda x: bool(x),
            validator=attrs.validators.instance_of(bool),
            kw_only=True,
            default=True,
        ),
        type="bool",
        init_type="bool",
        doc="Use multiple importance sampling. Default to True.",
    )

    @classmethod
    def from_onedim_to_grid(cls, phase: Abstract1DBlendPhaseFunction, xy_grid_shape):
        if len(xy_grid_shape) != 2:
            raise ValueError("The xy grid shape must be 2 dimensional")

        X, Y = xy_grid_shape

        weights = phase.weights
        weights = [[[c] * X] * Y for c in weights]

        return cls(
            weights=weights,
            components=phase.components,
            geometry=phase.geometry,
        )

    def _shape(self, arraylike: list) -> tuple:
        shape = []
        current = arraylike
        while isinstance(current, list):
            shape.append(len(current))
            current = current[0]
        return tuple(shape)

    @cache_by_id
    def _eval_weights_impl(self, si: SpectralIndex) -> np.ndarray:
        if isinstance(self.weights, list):
            weights_shape = self._shape(self.weights)
            assert len(weights_shape) == 3, f"Expected 3D weights, got {weights_shape}"
            weights = np.empty(weights_shape, dtype=np.float32)

            weights = None
            for c in range(weights_shape[0]):
                for x in range(weights_shape[1]):
                    for y in range(weights_shape[2]):
                        wsi = self.weights[c][x][y](si)
                        if weights is None:
                            weights = np.empty(
                                (*weights_shape, len(wsi)), dtype=np.float32
                            )
                        weights[c, x, y, :] = wsi

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
            n_component = range(len(self.components) - 1)
        elif isinstance(n_component, int):
            n_component = [n_component]

        return weights[n_component, ...]

    @property
    def template(self):
        if not isinstance(
            self.geometry, (type(None), PlaneParallelGeometry, SphericalShellGeometry)
        ):
            raise ValueError(
                f"unhandled scene geometry type '{type(self.geometry).__name__}'"
            )

        result = {"type": "multiphase", "use_mis": self.use_mis}

        for i in range(len(self.components)):
            template, _ = traverse(self.components[i])
            result.update(
                {f"phase_{i}.{k}": v for k, v in template.items()},
            )

            if self.geometry is None or isinstance(
                self.geometry, PlaneParallelGeometry
            ):
                # Note: This defines a partial and evaluates the component index.
                # Passing i as the kwarg default value is essential to force the
                # dereferencing of the loop variable.
                def eval_weights(ctx: KernelContext, n_component=i):
                    return mi.VolumeGrid(self._eval_weights(ctx.si, n_component).T)

                result[f"weight_{i}.type"] = "gridvolume"
                result[f"weight_{i}.grid"] = DictParameter(eval_weights)
                result[f"weight_{i}.filter_type"] = "nearest"

                if self.geometry is not None:
                    result[f"weight_{i}.to_world"] = (
                        self.geometry.atmosphere_volume_to_world
                    )

            elif isinstance(self.geometry, SphericalShellGeometry):
                # Same comment as above
                def eval_weights(ctx: KernelContext, n_component=i):
                    return mi.VolumeGrid(self._eval_weights(ctx.si, n_component))

                result[f"weight_{i}.type"] = "sphericalcoordsvolume"
                result[f"weight_{i}.volume.type"] = "gridvolume"
                result[f"weight_{i}.volume.grid"] = DictParameter(eval_weights)
                result[f"weight_{i}.volume.filter_type"] = "nearest"
                result[f"weight_{i}.to_world"] = (
                    self.geometry.atmosphere_volume_to_world
                )
                result[f"weight_{i}.rmin"] = self.geometry.atmosphere_volume_rmin

        return result

    @property
    def params(self) -> dict[str, SceneParameter]:
        result = {}

        if not isinstance(
            self.geometry, (type(None), PlaneParallelGeometry, SphericalShellGeometry)
        ):
            raise ValueError(
                f"unhandled scene geometry type '{type(self.geometry).__name__}'"
            )

        for i in range(len(self.components)):
            # Add components
            _, params = traverse(self.components[i])
            result.update({f"phase_{i}.{k}": v for k, v in params.items()})

            if self.geometry is None or isinstance(
                self.geometry, PlaneParallelGeometry
            ):

                def eval_weights(ctx: KernelContext, n_component=i):
                    return self._eval_weights(ctx.si, n_component).astype(np.float32).T

                result[f"weight_{i}.data"] = SceneParameter(
                    eval_weights,
                    KernelSceneParameterFlags.SPECTRAL,
                )

            elif isinstance(self.geometry, SphericalShellGeometry):

                def eval_weights(ctx: KernelContext, n_component=i):
                    return self._eval_weights(ctx.si, n_component).astype(np.float32)

                result[f"weight_{i}.volume.data"] = SceneParameter(
                    eval_weights,
                    KernelSceneParameterFlags.SPECTRAL,
                )

        return result

    def normalized(self, ctx: KernelContext):
        weights = self._eval_weights_impl(ctx.si)

        sum_weights = weights.sum(axis=0, keepdims=True)
        weights_normalized = weights / sum_weights

        return Multi3DPhaseFunction(
            components=self.components,
            weights=weights_normalized,
            geometry=self.geometry,
        )
