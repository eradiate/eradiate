from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from functools import singledispatchmethod

import numpy as np
import numpy.typing as npt
import pint
import pinttrs
from pinttrs.util import ensure_units

from . import CKDSpectralIndex, MonoSpectralIndex, SpectralIndex
from .ckd_quad import CKDQuadConfig, CKDQuadPolicy
from .response import BandSRF, DeltaSRF, SpectralResponseFunction, UniformSRF
from .. import converters
from .._mode import ModeFlag, SubtypeDispatcher
from ..attrs import define, documented
from ..constants import SPECTRAL_RANGE_MAX, SPECTRAL_RANGE_MIN
from ..quad import Quad
from ..radprops import AbsorptionDatabase, CKDAbsorptionDatabase, MonoAbsorptionDatabase
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg
from ..util.misc import summary_repr

# ------------------------------------------------------------------------------
#                             Class implementations
# ------------------------------------------------------------------------------


@define
class SpectralGrid(ABC):
    subtypes = SubtypeDispatcher("SpectralGrid")

    @property
    @abstractmethod
    def wavelengths(self):
        """
        Convenience accessor to characteristic wavelengths of this spectral grid.
        """
        pass

    @staticmethod
    def default() -> SpectralGrid:
        """
        Generate a default spectral grid depending on the active mode.
        """
        cls = SpectralGrid.subtypes.resolve()
        return cls.default()

    @staticmethod
    def from_absorption_database(abs_db: AbsorptionDatabase) -> SpectralGrid:
        """
        Retrieve the spectral grid from an absorption database. The returned
        type depends on the currently active mode.
        """
        cls = SpectralGrid.subtypes.resolve()
        return cls.from_absorption_database(abs_db)

    @abstractmethod
    def select(self, srf: SpectralResponseFunction) -> SpectralGrid:
        """
        Select a subset of the spectral grid based on a spectral response function.

        Parameters
        ----------
        srf : SpectralResponseFunction
            Spectral response function used to filter spectral grid points.

        Returns
        -------
        SpectralGrid
            New spectral grid instance covering the extent of the filtering SRF.

        Notes
        -----
        The implementation of this method uses single dispatch based on the type
        of the ``srf`` parameter.
        """
        pass

    @abstractmethod
    def walk_indices(self, **kwargs) -> t.Generator[SpectralIndex, None, None]:
        """
        A generator that yields a sequence of spectral index values.
        """
        pass


@SpectralGrid.subtypes.register(ModeFlag.SPECTRAL_MODE_MONO)
@define
class MonoSpectralGrid(SpectralGrid):
    _wavelengths: pint.Quantity = documented(
        pinttrs.field(
            units=ucc.deferred("wavelength"),
            converter=[
                pinttrs.converters.to_units(ucc.deferred("wavelength")),
                converters.on_quantity(np.atleast_1d),
                converters.on_quantity(lambda x: x.astype(np.float64)),
                converters.on_quantity(np.unique),
                converters.on_quantity(np.sort),
            ],
            repr=summary_repr,
        ),
        doc="Wavelengths.",
        type="quantity",
        init_type="quantity or array-like or float",
    )

    @property
    def wavelengths(self):
        return self._wavelengths

    @staticmethod
    def default() -> MonoSpectralGrid:
        """
        Generate a default monochromatic spectral grid that covers the default
        spectral range with 1 nm spacing.
        """
        return MonoSpectralGrid(
            wavelengths=np.arange(
                SPECTRAL_RANGE_MIN.m_as(ureg.nm),
                SPECTRAL_RANGE_MAX.m_as(ureg.nm) + 0.1,
                1.0,
            )
            * ureg.nm
        )

    @classmethod
    def from_absorption_database(cls, abs_db: MonoAbsorptionDatabase):
        """
        Retrieve the spectral grid from a monochromatic absorption database.
        """
        if not isinstance(abs_db, MonoAbsorptionDatabase):
            raise TypeError

        w = abs_db.spectral_coverage.index.get_level_values(level=1).values * ureg.nm
        return cls(wavelengths=w)

    @singledispatchmethod
    def select(self, srf: SpectralResponseFunction) -> MonoSpectralGrid:
        # Inherit docstring
        raise NotImplementedError(f"unsupported data type '{type(srf)}'")

    @select.register
    def _(self, srf: DeltaSRF):
        # Pass SRF wavelengths through
        return MonoSpectralGrid(wavelengths=srf.wavelengths)

    @select.register
    def _(self, srf: UniformSRF):
        w_m = self.wavelengths.m
        w_u = self.wavelengths.u
        wmin_m, wmax_m = srf.wmin.m_as(w_u), srf.wmax.m_as(w_u)

        w_selected_m = w_m[(w_m >= wmin_m) & (w_m <= wmax_m)]
        return MonoSpectralGrid(wavelengths=w_selected_m * w_u)

    @select.register
    def _(self, srf: BandSRF):
        # Select all wavelengths for which the SRF evaluates to a nonzero value
        values = srf.eval(self.wavelengths)
        w_selected = self.wavelengths[values.m > 0.0]
        return MonoSpectralGrid(wavelengths=w_selected)

    def walk_indices(self) -> t.Generator[MonoSpectralIndex, None, None]:
        # Inherit docstring
        for w in self.wavelengths:
            yield MonoSpectralIndex(w=w)


@SpectralGrid.subtypes.register(ModeFlag.SPECTRAL_MODE_CKD)
@define(init=False)
class CKDSpectralGrid(SpectralGrid):
    wmins: pint.Quantity = documented(
        pinttrs.field(units=ucc.deferred("wavelength"), repr=summary_repr),
        doc="Lower bound of all bins. Unitless values are interpreted as default "
        "wavelength config units.",
        type="quantity",
        init_type="quantity or array-like",
    )

    wmaxs: pint.Quantity = documented(
        pinttrs.field(units=ucc.deferred("wavelength"), repr=summary_repr),
        doc="Upper bound of all bins. Unitless values are interpreted as default "
        "wavelength config units.",
        type="quantity",
        init_type="quantity or array-like",
    )

    wcenters: pint.Quantity = documented(
        pinttrs.field(units=ucc.deferred("wavelength"), repr=summary_repr),
        doc="Central wavelength of all bins. Unitless values are interpreted as "
        "default wavelength config units. "
        "If unset, bin centers are computed automatically from bin bounds. "
        "Bin centers are allowed to be different from the middle of the bin "
        "interval and, when the grid is tied to a database, are expected to "
        "match the values of the wavelength coordinate in the database. However, "
        "this is considered as a workaround to deal with poorly indexed databases, "
        "and users should try to set central wavelengths to the middle of spectral "
        "bins.",
        type="quantity, optional",
        init_type="quantity or array-like",
    )

    def __init__(
        self,
        wmins: npt.ArrayLike,
        wmaxs: npt.ArrayLike,
        wcenters: npt.ArrayLike | None = None,
        fix_bounds: bool | t.Literal["keep_min", "keep_max"] = "keep_min",
        epsilon: float = 1e-6,
    ):
        # Ensure consistent units and appropriate dtype
        w_u = ucc.get("wavelength")
        wmins_m = ensure_units(wmins, w_u).m_as(w_u).astype(np.float64)
        wmaxs_m = ensure_units(wmaxs, w_u).m_as(w_u).astype(np.float64)

        # Detect bound mismatch
        diff_bounds = wmaxs_m[:-1] - wmins_m[1:]
        fix_locations = (diff_bounds > 0.0) & (diff_bounds <= epsilon)
        if np.any(fix_locations):
            if fix_bounds == "keep_max":
                wmins_m[1:] = np.where(fix_locations, wmaxs_m[:-1], wmins_m[1:])
            elif fix_bounds == "keep_min":
                wmaxs_m[:-1] = np.where(fix_locations, wmins_m[1:], wmaxs_m[:-1])
            else:
                raise ValueError(
                    "while constructing CKDSpectralGrid: bin bound mismatch "
                    f"(min: {wmins_m[1:][fix_locations]}; max: {wmaxs_m[:-1][fix_locations]}"
                )

        # Define bin centers if necessary
        if wcenters is None:
            wcenters = 0.5 * (wmins_m + wmaxs_m) * w_u

        # Initialize the object
        self.__attrs_init__(wmins_m * w_u, wmaxs_m * w_u, wcenters)

    @property
    def wavelengths(self):
        # Inherit docstring
        return self.wcenters

    @staticmethod
    def default() -> CKDSpectralGrid:
        """
        Generate a default CKD spectral that covers the default spectral range
        with 10 nm spacing.
        """
        return CKDSpectralGrid.arange(
            start=SPECTRAL_RANGE_MIN.m_as(ureg.nm),
            stop=SPECTRAL_RANGE_MAX.m_as(ureg.nm) + 1.0,
            step=10.0,
        )

    @classmethod
    def arange(
        cls,
        start: float | pint.Quantity,
        stop: float | pint.Quantity,
        step: float | pint.Quantity,
    ) -> CKDSpectralGrid:
        """
        Generate a CKD spectral grid with equally-sized bins.

        Parameters
        ----------
        start : quantity or float
            Central wavelength of the first bin. If a unitless value is passed,
            it is interpreted in default wavelength units (usually nm).

        stop : quantity or float
            Wavelength after which bin generation stops. If a unitless value is
            passed, it is interpreted in default wavelength units (usually nm).

        step : quantity or float
            Spectral bin size. If a unitless value is passed, it is interpreted
            in default wavelength units (usually nm).

        Returns
        -------
        CKDSpectralGrid
            Generated CKD spectral grid.
        """
        w_u = ucc.get("wavelength")
        start_m = ensure_units(start, w_u).m_as(w_u)
        stop_m = ensure_units(stop, w_u).m_as(w_u)
        width_m = ensure_units(step, w_u).m_as(w_u)

        wcenters_m = np.arange(start_m, stop_m + 0.1 * width_m, width_m)
        wmins_m = wcenters_m - 0.5 * width_m
        wmaxs_m = wcenters_m + 0.5 * width_m

        return cls(wmins_m * w_u, wmaxs_m * w_u, wcenters_m * w_u)

    @classmethod
    def from_nodes(cls, wnodes: npt.ArrayLike) -> CKDSpectralGrid:
        wmins = wnodes[:-1]
        wmaxs = wnodes[1:]
        return cls(wmins=wmins, wmaxs=wmaxs)

    @classmethod
    def from_absorption_database(cls, abs_db: CKDAbsorptionDatabase) -> CKDSpectralGrid:
        """
        Retrieve the spectral grid from a CKD absorption database.
        """
        if not isinstance(abs_db, CKDAbsorptionDatabase):
            raise TypeError

        wmins = abs_db.spectral_coverage["wbound_lower [nm]"].values * ureg.nm
        wmaxs = abs_db.spectral_coverage["wbound_upper [nm]"].values * ureg.nm
        wcenters = abs_db.spectral_coverage.index.get_level_values(1).values * ureg.nm
        return cls(wmins, wmaxs, wcenters)

    @singledispatchmethod
    def select(self, srf: SpectralResponseFunction) -> CKDSpectralGrid:
        # Inherit docstring
        raise NotImplementedError(f"unsupported data type '{type(srf)}'")

    @select.register
    def _(self, srf: DeltaSRF):
        w_u = srf.wavelengths.u
        w_m = srf.wavelengths.m
        wmins_m = self.wmins.m_as(w_u)
        wmaxs_m = self.wmaxs.m_as(w_u)

        selmin = np.searchsorted(wmins_m, w_m)
        selmax = np.searchsorted(wmaxs_m, w_m) + 1
        hit = selmin == selmax  # Mask where w_m values which triggered a bin hit

        # Map w values to selected bin (index -999 means not selected)
        bin_index = np.where(hit, selmin - 1, np.full_like(w_m, -999)).astype(np.int64)

        # Get selected bins only
        selected = np.unique(bin_index)  # mask removes -999 value
        selected = selected[selected >= 0]

        return CKDSpectralGrid(wmins=self.wmins[selected], wmaxs=self.wmaxs[selected])

    @select.register
    def _(self, srf: UniformSRF):
        selected = (self.wmaxs > srf.wmin) & (self.wmins < srf.wmax)
        return CKDSpectralGrid(wmins=self.wmins[selected], wmaxs=self.wmaxs[selected])

    @select.register
    def _(self, srf: BandSRF):
        w_u = self.wmins.u
        wmins_m = self.wmins.m_as(w_u)
        wmaxs_m = self.wmaxs.m_as(w_u)

        # Build spectral mesh used to interpolate
        w_m = np.unique(np.concatenate((wmins_m, wmaxs_m)))
        # Note the handling of numeric precision-induced min and max bin bound
        # mismatch was removed from the previous implementation because
        # consistency is enforced upon initialization

        # Detect spectral bins on which the SRF takes nonzero values
        cumsum = np.concatenate(([0], srf.integrate_cumulative(w_m * w_u).m_as(w_u)))
        selected = cumsum[:-1] != cumsum[1:]

        # Build a new spectral grid that only contains selected bins
        return CKDSpectralGrid(self.wmins[selected], self.wmaxs[selected])

    def walk_quads(
        self,
        ckd_quad_config: CKDQuadConfig,
        abs_db: CKDAbsorptionDatabase | None = None,
    ) -> t.Generator[Quad]:
        # Check parameter consistency
        if ckd_quad_config.policy is not CKDQuadPolicy.FIXED and abs_db is None:
            raise ValueError(
                "while attempting CKD spectral grid walk with policy "
                f"{ckd_quad_config.policy}: `abs_db` must be set (got None)"
            )

        # Walk the spectral grid and get the quadrature for each bin
        for w in self.wcenters:
            yield ckd_quad_config.get_quad(abs_db, wcenter=w)

    def walk_indices(
        self,
        ckd_quad_config: CKDQuadConfig,
        abs_db: CKDAbsorptionDatabase | None = None,
    ) -> t.Generator[CKDSpectralIndex]:
        # Inherit docstring

        quads = self.walk_quads(ckd_quad_config, abs_db)

        # Walk the spectral dimension
        for w in self.wcenters:
            quad = next(quads)
            for g in quad.eval_nodes([0, 1]):
                yield CKDSpectralIndex(w=w, g=g)
