from __future__ import annotations

import typing as t

import attrs
import mitsuba as mi
import numpy as np
import pint
import pinttrs

from ._core import PhaseFunction
from .._gridvolume import generate_gridvolume
from ..geometry import SceneGeometry
from ...attrs import define, documented
from ...contexts import KernelContext
from ...kernel import DictParameter, KernelSceneParameterFlags, SceneParameter
from ...spectral.index import SpectralIndex
from ...units import unit_context_config as ucc
from ...util.misc import cache_by_id


def _validate_increasing_grid_axis(instance, attribute, value):
    """Raise if ``value`` isn't a strictly increasing 1-D array."""
    if value.ndim != 1:
        raise ValueError(
            f"'{attribute.name}' must be a 1-D array, got shape {value.shape}"
        )

    if len(value) > 1 and np.any(np.diff(value) <= 0):
        raise ValueError(f"'{attribute.name}' must be strictly increasing")


@define(eq=False, slots=False)
class ParticleFieldPhaseFunction(PhaseFunction):
    """
    Phase function for spatially heterogeneous cloud particle fields.

    Wraps the ``particlefieldphase`` kernel plugin, which performs bilinear
    interpolation of the phase matrix in the ``(r_eff, v_eff)`` parameter
    space at each scattering interaction.

    The phase spectral properties are passed through the ``phase_data``
    callable. This callable returns a dict with the following information:

    * ``mu``: 2D array of shape
    ``(len(r_eff_grid) * len(v_eff_grid), max_len)``. Each row holds the
    scattering angle cosine discretization for one combination of an
    ``r_eff_grid`` value and a ``v_eff_grid`` value, padded with trailing NaN
    to the widest row's length.
    * ``phase``: 3D array of phase function values, shape
    ``(len(r_eff_grid) * len(v_eff_grid), n_phamat, max_len)``. Contains the
    tabulated phase function values indexed by ``mu``. It follows the same
    per-row grid-point ordering and NaN-padding as ``mu``, with an extra
    middle axis to select the Mueller phase matrix components.
    * ``ext``: extinction coefficient, shape ``(len(r_eff_grid), len(v_eff_grid))``.
    * ``ssa``: single-scattering albedo, shape  ``(len(r_eff_grid), len(v_eff_grid))``.
    * ``max_union_size``: upper bound on the CSR-flattened, cross-spectral
      angular index length built from ``mu``.
    """

    r_eff_volume: t.Callable[[KernelContext], pint.Quantity] = documented(
        attrs.field(kw_only=True),
        doc="Callable that returns a dense ``(n_x, n_y, n_z)`` quantity of "
        "effective radius values (length units) on the render grid for a "
        "given kernel context. NaN marks empty cells.",
        type="callable",
    )

    v_eff_volume: t.Callable[[KernelContext], pint.Quantity] = documented(
        attrs.field(kw_only=True),
        doc="Callable that returns a dense ``(n_x, n_y, n_z)`` quantity of "
        "effective variance values (dimensionless) on the render grid for "
        "a given kernel context. NaN marks empty cells.",
        type="callable",
    )

    r_eff_grid: pint.Quantity = documented(
        pinttrs.field(
            kw_only=True,
            units=ucc.deferred("length"),
            validator=[
                pinttrs.validators.has_compatible_units,
                _validate_increasing_grid_axis,
            ],
        ),
        doc="Strictly increasing 1-D array of effective radius values.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="quantity",
        init_type="array-like or quantity",
    )

    v_eff_grid: pint.Quantity = documented(
        pinttrs.field(
            kw_only=True,
            units=ucc.deferred("dimensionless"),
            validator=[
                pinttrs.validators.has_compatible_units,
                _validate_increasing_grid_axis,
            ],
        ),
        doc="Strictly increasing 1-D array of effective variance values.\n"
        "\n"
        "Unit-enabled field (default: ucc[dimensionless]).",
        type="quantity",
        init_type="array-like or quantity",
    )

    phase_data: t.Callable[[SpectralIndex], dict] = documented(
        attrs.field(kw_only=True),
        doc="Callable returning the phase properties for a given spectral "
        "index. Its return value is a dict with the keys ``mu``, ``phase``, "
        "``ext``, ``ssa`` and ``max_union_size``",
        type="callable",
    )

    geometry: SceneGeometry = documented(
        attrs.field(kw_only=True),
        doc="Geometry describing the render grid on which "
        "``r_eff_volume`` and ``v_eff_volume`` are evaluated when "
        "generating the kernel gridvolumes.",
        type="SceneGeometry",
    )

    @cache_by_id
    def _build_phase_parameters(self, si: SpectralIndex) -> dict:
        """
        Convert ``self.phase_data(si)`` into kernel-ready arrays: restructure
        the NaN-padded ``mu`` and ``phase`` arrays into the flat CSR form the
        ``particlephase`` kernel plugin expects, expand the reduced Mueller
        matrix representation, and derive the scattering weight
        (``ext * ssa``).

        Parameters
        ----------
        si : SpectralIndex
            Spectral index at which phase data is evaluated.

        Returns
        -------
        dict
            Kernel parameter dict with keys ``nodes``, ``phase_mueller``,
            ``grid_start`` and ``sigma_s_weight``, each a tensor. Used to
            configure the ``particlephase`` plugin.
        """
        data = self.phase_data(si)
        mu, phase, ext, ssa, max_union_size = (
            data["mu"],
            data["phase"],
            data["ext"],
            data["ssa"],
            data["max_union_size"],
        )

        n_phamat = phase.shape[1]

        if n_phamat == 4:
            expand_idx = [0, 1, 0, 2, 3, 2]
        elif n_phamat == 6:
            expand_idx = [0, 1, 2, 3, 4, 5]
        else:
            raise ValueError(f"unexpected 'phamat' size {n_phamat}")

        mueller = np.moveaxis(phase, 1, -1)[..., expand_idx]  # (n_pairs, max_len, 6)

        n_pairs = mu.shape[0]
        pair_lens = np.sum(~np.isnan(mu), axis=-1).astype(np.uint32)

        grid_start = np.zeros(n_pairs + 1, dtype=np.uint32)
        grid_start[1:] = np.cumsum(pair_lens)

        nodes = np.concatenate([mu[i, : pair_lens[i]] for i in range(n_pairs)])
        mueller_raw = np.concatenate(
            [mueller[i, : pair_lens[i]] for i in range(n_pairs)]
        )

        n_pad = max_union_size - len(nodes)
        if n_pad > 0:
            nodes = np.concatenate([nodes, np.full(n_pad, np.nan)])
            mueller_raw = np.concatenate(
                [mueller_raw, np.full((n_pad, mueller_raw.shape[-1]), np.nan)]
            )

        sigma_s_weight = ext * ssa

        return {
            "nodes": mi.TensorXf(nodes.reshape(-1, 1)),
            "phase_mueller": mi.TensorXf(mueller_raw),
            "grid_start": mi.TensorXu(grid_start.reshape(-1, 1)),
            "sigma_s_weight": mi.TensorXf(sigma_s_weight.reshape(-1, 1)),
        }

    def _eval_param(self, key: str):
        """
        Return a closure that, given a :class:`.KernelContext`, extracts
        ``key`` from :meth:`_build_phase_parameters`'s result.
        """

        def eval_param(ctx):
            return self._build_phase_parameters(ctx.si)[key]

        return eval_param

    @property
    def template(self) -> dict:
        return {
            "type": "particlephase",
            "r_eff_volume": generate_gridvolume(
                self.geometry,
                self.r_eff_volume,
                units=self.r_eff_grid.units,
                dtype=np.float64,
            ),
            "v_eff_volume": generate_gridvolume(
                self.geometry,
                self.v_eff_volume,
                units=self.v_eff_grid.units,
                dtype=np.float64,
            ),
            "r_eff_grid": mi.TensorXf(
                self.r_eff_grid.m_as(self.r_eff_grid.units).reshape(-1, 1)
            ),
            "v_eff_grid": mi.TensorXf(
                self.v_eff_grid.m_as(self.v_eff_grid.units).reshape(-1, 1)
            ),
            "nodes": DictParameter(self._eval_param("nodes")),
            "phase_mueller": DictParameter(self._eval_param("phase_mueller")),
            "grid_start": DictParameter(self._eval_param("grid_start")),
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
                self._eval_param("grid_start"),
                KernelSceneParameterFlags.SPECTRAL,
            ),
            "sigma_s_weight": SceneParameter(
                self._eval_param("sigma_s_weight"), KernelSceneParameterFlags.SPECTRAL
            ),
        }
