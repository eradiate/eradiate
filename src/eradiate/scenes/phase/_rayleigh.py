from __future__ import annotations

import typing as t

import attrs
import mitsuba as mi
import numpy as np

import eradiate

from ._core import PhaseFunction
from ..geometry import PlaneParallelGeometry, SceneGeometry, SphericalShellGeometry
from ...attrs import define, documented
from ...contexts import KernelContext
from ...kernel import DictParameter, KernelSceneParameterFlags, SceneParameter
from ...spectral.index import SpectralIndex


@define(eq=False, slots=False)
class RayleighPhaseFunction(PhaseFunction):
    """
    Rayleigh phase function [``rayleigh``].

    The Rayleigh phase function models scattering by particles with a
    characteristic size much smaller than the considered radiation wavelength.
    It is typically used to represent scattering by gas molecules in the
    atmosphere.
    """

    depolarization: np.ndarray | t.Callable[[KernelContext], np.ndarray] = documented(
        attrs.field(
            converter=lambda x: x if callable(x) else np.array(x, dtype=np.float64),
            kw_only=True,
            factory=lambda: np.array([0.0]),
        ),
        doc="The ratio of intensities parallel and perpendicular to the "
        "plane of scattering for light scattered at 90 deg. Only relevant "
        "when using a polarization mode.",
        type="ndarray, callable or None",
        init_type="array-like, callable, or None",
    )

    geometry: SceneGeometry | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(SceneGeometry.convert),
            validator=attrs.validators.optional(
                attrs.validators.instance_of(SceneGeometry)
            ),
        ),
        doc="Parameters defining the basic geometry of the scene. If unset, "
        "the volume textures defining component weights will be assigned "
        "defaults likely unsuitable for atmosphere construction.",
        type=".SceneGeometry or None",
        init_type=".SceneGeometry or dict or str, optional",
        default="None",
    )

    def _eval_depolarization_factor_impl(self, si: SpectralIndex) -> np.ndarray:
        if isinstance(self.depolarization, t.Callable):
            depolarization = self.depolarization(si)
        elif isinstance(self.depolarization, np.ndarray):
            depolarization = np.atleast_1d(self.depolarization)
        elif self.depolarization is None:
            depolarization = np.zeros((1,))
        else:
            NotImplementedError

        return depolarization

    def eval_depolarization_factor(self, si: SpectralIndex) -> np.ndarray:
        """
        Evaluate the depolarization factor.

        Parameters
        ----------
        si: .SpectralIndex
            Spectral context.

        Returns
        -------
            Depolarization factor as an array of shape (N) where N can
            either be 1 or the number of layers of the atmosphere's vertical
            axis.
        """
        depolarization = self._eval_depolarization_factor_impl(si)
        return depolarization

    @property
    def template(self) -> dict:
        if eradiate.mode().is_polarized:
            result = {"type": "rayleigh_polarized"}

            if self.geometry is None or isinstance(
                self.geometry, PlaneParallelGeometry
            ):
                if self.geometry is None:
                    to_world = mi.ScalarTransform4f()
                else:
                    to_world = self.geometry.atmosphere_volume_to_world

                result["depolarization.type"] = "gridvolume"
                result["depolarization.grid"] = DictParameter(
                    lambda ctx: mi.VolumeGrid(
                        np.reshape(
                            self.eval_depolarization_factor(ctx.si),
                            (-1, 1, 1),
                        ).astype(np.float32),
                    ),
                )
                result["depolarization.to_world"] = to_world
                result["depolarization.filter_type"] = "nearest"

            elif isinstance(self.geometry, SphericalShellGeometry):
                volume_rmin = self.geometry.atmosphere_volume_rmin
                to_world = self.geometry.atmosphere_volume_to_world

                result["depolarization.type"] = "sphericalcoordsvolume"
                result["depolarization.volume.type"] = "gridvolume"
                result["depolarization.volume.grid"] = DictParameter(
                    lambda ctx: mi.VolumeGrid(
                        np.reshape(
                            self.eval_depolarization_factor(ctx.si),
                            (-1, 1, 1),
                        ).astype(np.float32),
                    ),
                )
                result["depolarization.volume.filter_type"] = "nearest"
                result["depolarization.to_world"] = to_world
                result["depolarization.rmin"] = volume_rmin

            return result
        else:
            return {"type": "rayleigh"}

    @property
    def params(self) -> dict[str, SceneParameter]:
        result = {}

        if eradiate.mode().is_polarized:
            if self.geometry is None or isinstance(
                self.geometry, PlaneParallelGeometry
            ):
                result["depolarization.data"] = SceneParameter(
                    lambda ctx: np.reshape(
                        self.eval_depolarization_factor(ctx.si),
                        (-1, 1, 1, 1),
                    ).astype(np.float32),
                    KernelSceneParameterFlags.SPECTRAL,
                    # search=SearchSceneParameter(
                    #     node_type=mi.PhaseFunction,
                    #     node_id=self.phase.id,
                    #     parameter_relpath=,
                    # ),
                )

            elif isinstance(self.geometry, SphericalShellGeometry):
                result["depolarization.volume.data"] = SceneParameter(
                    lambda ctx: np.reshape(
                        self.eval_depolarization_factor(ctx.si),
                        (1, 1, -1, 1),
                    ).astype(np.float32),
                    KernelSceneParameterFlags.SPECTRAL,
                    # search=SearchSceneParameter(
                    #     node_type=mi.PhaseFunction,
                    #     node_id=self.phase.id,
                    #     parameter_relpath=,
                    # ),
                )
        return result
