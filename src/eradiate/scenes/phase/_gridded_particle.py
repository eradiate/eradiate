from __future__ import annotations

from typing import Callable

import attrs
import mitsuba as mi
import numpy as np
import pint

from ._core import PhaseFunction
from .._gridvolume import generate_gridvolume
from ..geometry import SceneGeometry
from ...attrs import define, documented
from ...kernel import DictParameter, KernelSceneParameterFlags, SceneParameter
from ...spectral.index import SpectralIndex
from ...units import unit_registry as ureg
from ...units import units_compatible
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
class GriddedParticlePhaseFunction(PhaseFunction):
    """
    Phase function for spatially heterogeneous particle components, looked
    up per voxel from one or two state variables.

    Wraps the ``particlefieldphase`` kernel plugin, which performs bilinear
    interpolation of the phase matrix over its two interpolation slots. When
    only one state variable is used, the second slot is padded with a
    degenerate, single-point grid, over which interpolation is a no-op.

    The phase spectral properties are passed through the ``phase_data``
    callable. This callable returns a dict with the following information:

    * ``mu``: 2D array of shape ``(n_pts, max_len)``, where ``n_pts`` is
      ``len(grid_1)`` (or ``len(grid_1) * len(grid_2)`` if a second state
      variable is used). Each row holds the scattering angle cosine
      discretization for one grid point, padded with trailing NaN to the
      widest row's length.
    * ``phase``: 3D array of phase function values, shape
      ``(n_pts, n_phamat, max_len)``. Contains the tabulated phase function
      values indexed by ``mu``. It follows the same per-row grid-point
      ordering and NaN-padding as ``mu``, with an extra middle axis to
      select the Mueller phase matrix components.
    * ``ext``: extinction coefficient, shape matching the state variable
      grid (``(len(grid_1),)`` or ``(len(grid_1), len(grid_2))``).
    * ``ssa``: single-scattering albedo, same shape as ``ext``.
    * ``max_union_size``: upper bound on the CSR-flattened, cross-spectral
      angular index length built from ``mu``.
    """

    volume_1: pint.Quantity = documented(
        attrs.field(kw_only=True),
        doc="Dense ``(n_x, n_y, n_z)`` quantity of the first state "
        "variable on the render grid. NaN marks empty cells.",
        type="quantity",
    )

    grid_1: pint.Quantity = documented(
        attrs.field(kw_only=True, validator=_validate_increasing_grid_axis),
        doc="Strictly increasing 1-D array of values for the first state "
        "variable. Must have units compatible with :attr:`volume_1`.",
        type="quantity",
        init_type="quantity",
    )

    @grid_1.validator
    def _validate_grid_1_volume_1(self, attribute, value):
        if not units_compatible(self.volume_1.units, value.units):
            raise ValueError(
                "'grid_1' and 'volume_1' must have compatible units, got "
                f"'{value.units}' and '{self.volume_1.units}'"
            )

    volume_2: pint.Quantity | None = documented(
        attrs.field(kw_only=True, default=None),
        doc="Dense ``(n_x, n_y, n_z)`` quantity of the second state "
        "variable on the render grid, if any. NaN marks empty cells.",
        type="quantity or None",
        init_type="quantity, optional",
        default="None",
    )

    grid_2: pint.Quantity | None = documented(
        attrs.field(
            kw_only=True,
            default=None,
            validator=attrs.validators.optional(_validate_increasing_grid_axis),
        ),
        doc="Strictly increasing 1-D array of values for the second state "
        "variable, if any. Must be set iff :attr:`volume_2` is set, with "
        "compatible units.",
        type="quantity or None",
        init_type="quantity, optional",
        default="None",
    )

    @grid_2.validator
    def _validate_grid_2_volume_2(self, attribute, value):
        if (value is None) != (self.volume_2 is None):
            raise ValueError("'grid_2' and 'volume_2' must be both set or both unset")

        if value is not None and not units_compatible(self.volume_2.units, value.units):
            raise ValueError(
                "'grid_2' and 'volume_2' must have compatible units, got "
                f"'{value.units}' and '{self.volume_2.units}'"
            )

    var_1: str = documented(
        attrs.field(kw_only=True, default="r_eff"),
        doc="Name of the state variable held by :attr:`volume_1`/"
        ":attr:`grid_1`. Used in the kernel dict generation.",
        type="str",
        init_type="str, optional",
        default='"r_eff"',
    )

    var_2: str = documented(
        attrs.field(kw_only=True, default="v_eff"),
        doc="Name of the state variable held by :attr:`volume_2`/"
        ":attr:`grid_2`, if set. Used in the kernel dict generation",
        type="str",
        init_type="str, optional",
        default='"v_eff"',
    )

    phase_data: Callable[[SpectralIndex], dict] = documented(
        attrs.field(kw_only=True),
        doc="Callable returning the phase properties for a given spectral "
        "index. Its return value is a dict with the keys ``mu``, ``phase``, "
        "``ext``, ``ssa`` and ``max_union_size``",
        type="callable",
    )

    geometry: SceneGeometry = documented(
        attrs.field(kw_only=True),
        doc="Geometry describing the render grid on which "
        "``volume_1`` and ``volume_2`` are defined.",
        type="SceneGeometry",
    )

    def _grid_2_or_default(self) -> pint.Quantity:
        """Return :attr:`grid_2`, or a degenerate single-point placeholder."""
        if self.grid_2 is not None:
            return self.grid_2
        return ureg.Quantity([0.0], ureg.dimensionless)

    def _volume_2_or_default(self) -> pint.Quantity:
        """Return :attr:`volume_2`, or a degenerate constant placeholder."""
        if self.volume_2 is not None:
            return self.volume_2
        return ureg.Quantity(0.0, ureg.dimensionless)

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

        sigma_s_weight = (ext * ssa).m

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
        grid_2 = self._grid_2_or_default()
        volume_2 = self._volume_2_or_default()
        # The plugin only interpolates volume values against grid nodes, so any
        # unit shared by a grid and its volume works: use the grid's own
        unit_1 = self.grid_1.units
        unit_2 = grid_2.units

        return {
            "type": "particlephase",
            f"{self.var_1}_volume": generate_gridvolume(
                self.geometry,
                lambda ctx: self.volume_1,
                units=unit_1,
                dtype=np.float64,
            ),
            f"{self.var_2}_volume": generate_gridvolume(
                self.geometry,
                lambda ctx: volume_2,
                units=unit_2,
                dtype=np.float64,
            ),
            f"{self.var_1}_grid": mi.TensorXf(self.grid_1.m_as(unit_1).reshape(-1, 1)),
            f"{self.var_2}_grid": mi.TensorXf(grid_2.m_as(unit_2).reshape(-1, 1)),
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
