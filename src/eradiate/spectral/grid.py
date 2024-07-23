from __future__ import annotations

from abc import ABC, abstractmethod
from functools import singledispatchmethod

import numpy as np
import pint
import pinttrs

from eradiate.attrs import define, documented

from .response import BandSRF, DeltaSRF, SpectralResponseFunction, UniformSRF
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
    def _(self, srf: UniformSRF) -> MonoSpectralGrid:
        w_m = self.wavelengths.m
        w_u = self.wavelengths.u
        wmin_m, wmax_m = srf.wmin.m_as(w_u), srf.wmax.m_as(w_u)

        w_selected_m = w_m[(w_m >= wmin_m) & (w_m <= wmax_m)]
        return MonoSpectralGrid(wavelengths=w_selected_m * w_u)

    @select.register
    def _(self, srf: BandSRF) -> MonoSpectralGrid:
        w_m = self.wavelengths.m
        w_u = self.wavelengths.u
        wmin_m, wmax_m = srf.support().m_as(w_u)

        w_selected_m = w_m[(w_m >= wmin_m) & (w_m <= wmax_m)]
        return MonoSpectralGrid(wavelengths=w_selected_m * w_u)


@define
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

    def wavelengths(self):
        # Inherit docstring
        return self.wcenters

    @classmethod
    def from_nodes(cls, wnodes):
        wmins = wnodes[:-1]
        wmaxs = wnodes[1:]
        wcenters = 0.5 * (wmins + wmaxs)
        return cls(wcenters=wcenters, wmins=wmins, wmaxs=wmaxs)

    @classmethod
    def from_absorption_database(cls, abs_db: CKDAbsorptionDatabase):
        raise NotImplementedError

    @singledispatchmethod
    def select(self, srf: SpectralResponseFunction) -> CKDSpectralGrid:
        # Inherit docstring
        raise NotImplementedError(f"unsupported data type '{type(srf)}'")

    @select.register
    def _(self, srf: DeltaSRF) -> CKDSpectralGrid:
        raise NotImplementedError

    @select.register
    def _(self, srf: UniformSRF) -> CKDSpectralGrid:
        raise NotImplementedError

    @select.register
    def _(self, srf: BandSRF) -> CKDSpectralGrid:
        # Selection rationale:
        # 1. Detect spectral bins on which the SRF takes nonzero values
        # 2. Select those bins
        bins = binset.bins
        wunits = "nm"
        xmin = np.array([bin.wmin.m_as(wunits) for bin in bins])
        xmax = np.array([bin.wmax.m_as(wunits) for bin in bins])
        r = self.values.m
        w = self.wavelengths.m_as(wunits)

        # Evaluate the SRF on the bin grid
        resolution = (max(xmax) - min(xmax)) / (len(xmax) - 1)
        epsilon = resolution * 1e-3
        bins = np.unique((xmin, xmax))
        precision_mask = np.ones((len(bins)), dtype=bool)
        precision_mask[1:] = np.abs(np.diff(bins)) > epsilon
        bins = bins[precision_mask]
        srf_bins = np.interp(bins, w, srf, left=0, right=0)
        return np.where(nonzero_integral(bins, srf_bins))[0]

        return BinSet(bins=list(np.array(bins)[selected]))
