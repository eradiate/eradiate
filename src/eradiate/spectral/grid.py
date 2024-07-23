from __future__ import annotations

from abc import ABC, abstractmethod
from functools import singledispatchmethod

import numpy as np
import numpy.typing as npt
import pint
import pinttrs
from pinttrs.util import ensure_units

from .response import BandSRF, DeltaSRF, SpectralResponseFunction, UniformSRF
from ..attrs import define, documented
from ..radprops import CKDAbsorptionDatabase
from ..units import unit_context_config as ucc


@define
class SpectralGrid(ABC):
    @property
    @abstractmethod
    def wavelengths(self):
        """
        Convenience accessor to characteristic wavelengths of this spectral grid.
        """
        pass

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


@define
class MonoSpectralGrid(SpectralGrid):
    _wavelengths: pint.Quantity = documented(
        pinttrs.field(
            units=ucc.deferred("wavelength"),
            on_setattr=None,
            converter=np.atleast_1d,
        ),
        doc="Wavelengths.",
        type="quantity",
        init_type="quantity or array-like or float",
    )

    @property
    def wavelengths(self):
        return self._wavelengths

    @singledispatchmethod
    def select(self, srf: SpectralResponseFunction) -> MonoSpectralGrid:
        # Inherit docstring
        raise NotImplementedError(f"unsupported data type '{type(srf)}'")

    @select.register
    def _(self, srf: UniformSRF):
        w_m = self.wavelengths.m
        w_u = self.wavelengths.u
        wmin_m, wmax_m = srf.wmin.m_as(w_u), srf.wmax.m_as(w_u)

        w_selected_m = w_m[(w_m >= wmin_m) & (w_m <= wmax_m)]
        return MonoSpectralGrid(wavelengths=w_selected_m * w_u)

    @select.register
    def _(self, srf: BandSRF):
        w_m = self.wavelengths.m
        w_u = self.wavelengths.u
        wmin_m, wmax_m = srf.support().m_as(w_u)

        w_selected_m = w_m[(w_m >= wmin_m) & (w_m <= wmax_m)]
        return MonoSpectralGrid(wavelengths=w_selected_m * w_u)


@define(init=False)
class CKDSpectralGrid(SpectralGrid):
    wcenters: pint.Quantity = documented(
        pinttrs.field(units=ucc.deferred("wavelength")),
        doc="Central wavelength of all bins.",
        type="quantity",
        init_type="quantity or array-like or float",
    )

    wmins: pint.Quantity = documented(
        pinttrs.field(units=ucc.deferred("wavelength")),
        doc="Lower bound of all bins.",
        type="quantity",
        init_type="quantity or array-like or float",
    )

    wmaxs: pint.Quantity = documented(
        pinttrs.field(units=ucc.deferred("wavelength")),
        doc="Upper bound of all bins.",
        type="quantity",
        init_type="quantity or array-like or float",
    )

    def __init__(self, wmins: npt.ArrayLike, wmaxs: npt.ArrayLike):
        w_u = ucc.get("wavelength")
        wmins_m = ensure_units(wmins, w_u).m_as(w_u)
        wmaxs_m = ensure_units(wmaxs, w_u).m_as(w_u)
        wcenters_m = 0.5 * (wmins_m + wmaxs_m)
        self.__attrs_init__(wcenters_m * w_u, wmins_m * w_u, wmaxs_m * w_u)

    def wavelengths(self):
        # Inherit docstring
        return self.wcenters

    @classmethod
    def arange(
        cls, start: npt.ArrayLike, stop: npt.ArrayLike, step: float | pint.Quantity
    ) -> CKDSpectralGrid:
        w_u = ucc.get("wavelength")
        start_m = ensure_units(start, w_u).m_as(w_u)
        stop_m = ensure_units(stop, w_u).m_as(w_u)
        width_m = ensure_units(step, w_u).m_as(w_u)

        wcenters_m = np.arange(start_m, stop_m + 0.1 * width_m, width_m)
        wmins_m = wcenters_m - 0.5 * width_m
        wmaxs_m = wcenters_m + 0.5 * width_m

        return cls(wmins_m * w_u, wmaxs_m * w_u)

    @classmethod
    def from_nodes(cls, wnodes: npt.ArrayLike):
        wmins = wnodes[:-1]
        wmaxs = wnodes[1:]
        return cls(wmins=wmins, wmaxs=wmaxs)

    @classmethod
    def from_absorption_database(cls, abs_db: CKDAbsorptionDatabase):
        raise NotImplementedError

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
        hit = selmin == selmax  # Mask where x values which triggered a bin hit

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

        # Detect spectral bins on which the SRF takes nonzero values
        # Selected bins are identified by their central wavelength
        w_m = np.unique(np.concatenate((wmins_m, wmaxs_m)))
        cumsum = np.concatenate(([0], srf.integrate_cumulative(w_m * w_u).m_as(w_u)))
        selected = cumsum[:-1] != cumsum[1:]

        # Build a new spectral grid that only contains selected bins
        return CKDSpectralGrid(self.wmins[selected], self.wmaxs[selected])
