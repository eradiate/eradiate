from __future__ import annotations

import typing as t

import attrs
import drjit as dr
import mitsuba as mi
import numpy as np

from ._core import PhaseFunction
from .._gridvolume import generate_gridvolume
from ..geometry import SceneGeometry
from ...attrs import define, documented
from ...contexts import KernelContext
from ...kernel import DictParameter, KernelSceneParameterFlags, SceneParameter
from ...spectral.index import SpectralIndex
from ...units import unit_registry as ureg
from ...util.misc import cache_by_id


def _validate_regular_grid_axis(instance, attribute, value):
    """Raise if ``value`` isn't a sorted, regularly-spaced 1-D array."""
    value = np.asarray(value)

    if value.ndim != 1:
        raise ValueError(
            f"'{attribute.name}' must be a 1-D array, got shape {value.shape}"
        )

    if len(value) > 1:
        steps = np.diff(value)
        if np.any(steps <= 0):
            raise ValueError(f"'{attribute.name}' must be strictly increasing")
        if not np.allclose(steps, steps[0]):
            raise ValueError(f"'{attribute.name}' must be regularly spaced")


@define(eq=False, slots=False)
class ParticleFieldPhaseFunction(PhaseFunction):
    """
    Phase function for spatially heterogeneous cloud particle fields.

    Wraps the ``particlefieldphase`` kernel plugin, which performs bilinear
    interpolation of the phase matrix in the ``(r_eff, v_eff)`` parameter
    space at each scattering interaction.
    """

    r_eff_volume: t.Callable[[KernelContext], np.ndarray] = documented(
        attrs.field(kw_only=True),
        doc="Callable that returns a dense ``(n_x, n_y, n_z)`` array of "
        "effective radius values on the render grid for a given kernel "
        "context. NaN marks empty cells.",
        type="callable",
    )

    v_eff_volume: t.Callable[[KernelContext], np.ndarray] = documented(
        attrs.field(kw_only=True),
        doc="Callable that returns a dense ``(n_x, n_y, n_z)`` array of "
        "effective variance values on the render grid for a given kernel "
        "context. NaN marks empty cells.",
        type="callable",
    )

    r_eff_grid: np.ndarray = documented(
        attrs.field(kw_only=True, validator=_validate_regular_grid_axis),
        doc="Sorted, regularly-spaced 1-D array of effective radius values "
        "in the properties dataset.",
        type="ndarray",
    )

    v_eff_grid: np.ndarray = documented(
        attrs.field(kw_only=True, validator=_validate_regular_grid_axis),
        doc="Sorted, regularly-spaced 1-D array of effective variance "
        "values in the properties dataset.",
        type="ndarray",
    )

    phase_data: t.Callable[
        [SpectralIndex],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = documented(
        attrs.field(kw_only=True),
        doc="Callable that returns ``(mu, phase, nangles, ext, ssa)`` for a "
        "spectral index. ``mu`` and ``phase`` are NaN-padded along their "
        "last axis to the largest angle count found across all "
        "``(r_eff, v_eff)`` pairs. ``nangles`` gives the actual valid "
        "angle count for each pair. ``ext`` and ``ssa`` have shape "
        "``(n_r, n_v)``.",
        type="callable",
    )

    geometry: SceneGeometry = documented(
        attrs.field(kw_only=True),
        doc="Geometry describing the render grid on which "
        "``r_eff_volume`` and ``v_eff_volume`` are evaluated when "
        "generating the kernel gridvolumes.",
        type="SceneGeometry",
    )

    blending_method: str = documented(
        attrs.field(default="search", kw_only=True),
        doc="Phase blending method passed to the kernel plugin. "
        'Either ``"search"``, ``"stochastic"`` or ``"tabulate"``.',
        type="str",
        default='"search"',
    )

    sigma_s_override: float | None = documented(
        attrs.field(default=None, kw_only=True),
        doc="Debugging switch. If set, overrides the per-``(r_eff, v_eff)`` "
        "scattering coefficient weight (``ext * ssa``) used to blend phase "
        "matrices, instead of the value derived from ``phase_data``.",
        type="float or None",
        init_type="float, optional",
        default="None",
    )

    @cache_by_id
    def _build_phase_parameters(self, si: SpectralIndex) -> dict:
        """
        Convert ``self.phase_data(si)`` into flat kernel-ready arrays: strip
        NaN padding and concatenate each ``(r_eff, v_eff)`` pair's own valid
        angles, expand the reduced Mueller matrix representation, and derive
        the per-pair scattering weight (``ext * ssa``, optionally overridden
        by ``sigma_s_override``).

        Parameters
        ----------
        si : SpectralIndex
            Spectral index at which phase data is evaluated.

        Returns
        -------
        dict
            Kernel parameter dict with keys ``r_eff_grid``, ``v_eff_grid``,
            ``nodes``, ``phase_mueller``, ``grid_start``, ``grid_len`` and
            ``sigma_s_weight``.
        """
        mu, phase, nangles, ext, ssa = self.phase_data(si)

        n_r, n_v = ext.shape
        n_phamat = phase.shape[0]
        n_pts = n_r * n_v

        if n_phamat == 4:
            expand_idx = [0, 1, 0, 2, 3, 2]
        elif n_phamat == 6:
            expand_idx = [0, 1, 2, 3, 4, 5]
        else:
            raise ValueError(f"unexpected 'phamat' size {n_phamat}")

        mu_flat = mu.reshape(n_pts, -1)
        phase_flat = phase.reshape(n_phamat, n_pts, -1)
        grid_len = nangles.reshape(n_pts).astype(np.uint32)
        grid_start = np.zeros(n_pts, dtype=np.uint32)
        grid_start[1:] = np.cumsum(grid_len)[:-1]

        # Strip nan padding and concatenate each pair's own valid angles.
        nodes_raw = np.concatenate(
            [mu_flat[i, : grid_len[i]] for i in range(n_pts)]
        ).astype(np.float64)

        mueller_raw = np.concatenate(
            [
                np.moveaxis(phase_flat[:, i, : grid_len[i]], 0, -1)[..., expand_idx]
                for i in range(n_pts)
            ],
            axis=0,
        ).astype(np.float64)

        sigma_s_weight = (ext * ssa).flatten().astype(np.float64)

        if self.sigma_s_override is not None:
            sigma_s_weight[:] = self.sigma_s_override

        return {
            "r_eff_grid": self.r_eff_grid.astype(np.float64),
            "v_eff_grid": self.v_eff_grid.astype(np.float64),
            "nodes": dr.scalar.ArrayXf64(nodes_raw),
            "phase_mueller": dr.scalar.ArrayXf64(mueller_raw.flatten()),
            "grid_start": grid_start,
            "grid_len": grid_len,
            "sigma_s_weight": dr.scalar.ArrayXf64(sigma_s_weight),
        }

    def _eval_param(self, key: str, wrap: t.Callable | None = None):
        """
        Return a closure that, given a :class:`.KernelContext`, extracts
        ``key`` from :meth:`_build_phase_parameters`'s result, optionally
        passing it through ``wrap``.
        """

        def eval_param(ctx):
            value = self._build_phase_parameters(ctx.si)[key]
            return value if wrap is None else wrap(value)

        return eval_param

    @property
    def template(self) -> dict:
        return {
            "type": "particlefieldphase",
            "r_eff_volume": generate_gridvolume(
                self.geometry,
                self.r_eff_volume,
                units=ureg.micron,
                dtype=np.float64,
            ),
            "v_eff_volume": generate_gridvolume(
                self.geometry,
                self.v_eff_volume,
                units=ureg.dimensionless,
                dtype=np.float64,
            ),
            "r_eff_grid": mi.VolumeGrid(
                self.r_eff_grid.astype(np.float64).reshape(1, 1, -1, 1)
            ),
            "v_eff_grid": mi.VolumeGrid(
                self.v_eff_grid.astype(np.float64).reshape(1, 1, -1, 1)
            ),
            "nodes": DictParameter(self._eval_param("nodes")),
            "phase_mueller": DictParameter(self._eval_param("phase_mueller")),
            "grid_start": DictParameter(
                self._eval_param("grid_start", dr.scalar.ArrayXu)
            ),
            "grid_len": DictParameter(self._eval_param("grid_len", dr.scalar.ArrayXu)),
            "blending_method": self.blending_method,
            "sigma_s_weight": DictParameter(self._eval_param("sigma_s_weight")),
        }

    @property
    def params(self) -> dict[str, SceneParameter]:
        return {
            "nodes": SceneParameter(
                self._eval_param("nodes"), KernelSceneParameterFlags.SPECTRAL
            ),
            "phase_mueller": SceneParameter(
                self._eval_param("phase_mueller"), KernelSceneParameterFlags.SPECTRAL
            ),
            "grid_start": SceneParameter(
                self._eval_param("grid_start", dr.scalar.ArrayXu),
                KernelSceneParameterFlags.SPECTRAL,
            ),
            "grid_len": SceneParameter(
                self._eval_param("grid_len", dr.scalar.ArrayXu),
                KernelSceneParameterFlags.SPECTRAL,
            ),
            "sigma_s_weight": SceneParameter(
                self._eval_param("sigma_s_weight"), KernelSceneParameterFlags.SPECTRAL
            ),
        }
