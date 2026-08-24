from __future__ import annotations

import typing as t

import attrs
import numpy as np

import eradiate

from ._core import PhaseFunction
from .._gridvolume import generate_gridvolume
from ..geometry import PlaneParallelGeometry, SceneGeometry, SphericalShellGeometry
from ...attrs import define, documented
from ...contexts import KernelContext
from ...kernel import KernelSceneParameterFlags, SceneParameter
from ...spectral.index import SpectralIndex
from ...util.misc import cache_by_id


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
        "the volume textures defining component depolarization factor will be "
        "assigned defaults likely unsuitable for atmosphere construction.",
        type=".SceneGeometry or None",
        init_type=".SceneGeometry or dict or str, optional",
        default="None",
    )

    @cache_by_id
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
        if isinstance(self.depolarization, t.Callable):
            depolarization = self.depolarization(si)
        elif isinstance(self.depolarization, np.ndarray):
            depolarization = np.atleast_1d(self.depolarization)
        elif self.depolarization is None:
            depolarization = np.zeros((1,))
        else:
            raise NotImplementedError()

        return depolarization

    @property
    def template(self) -> dict:
        if eradiate.mode().is_polarized:
            depolarization_grid = generate_gridvolume(
                self.geometry,
                self.eval_depolarization_factor,
            )

            return {
                "type": "rayleigh_polarized",
                "depolarization": depolarization_grid,
            }

        else:
            return {"type": "rayleigh"}

    @property
    def params(self) -> dict[str, SceneParameter]:
        result = {}

        if eradiate.mode().is_polarized:
            depolarization_grid = generate_gridvolume(
                self.geometry,
                self.eval_depolarization_factor,
                flag=KernelSceneParameterFlags.SPECTRAL,
            )

            if self.geometry is None or isinstance(
                self.geometry, PlaneParallelGeometry
            ):
                result["depolarization.data"] = depolarization_grid
            elif isinstance(self.geometry, SphericalShellGeometry):
                result["depolarization.volume.data"] = depolarization_grid

        return result
