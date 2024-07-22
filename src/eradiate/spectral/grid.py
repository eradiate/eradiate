from __future__ import annotations

from abc import ABC, abstractmethod
from functools import singledispatchmethod

import numpy as np
import pint
import pinttrs

from eradiate.attrs import define, documented

from .response import BandSRF, SpectralResponseFunction, UniformSRF
from ..radprops import CKDAbsorptionDatabase
from ..units import unit_context_config as ucc


@define
class SpectralGrid(ABC):
    @property
    @abstractmethod
    def wavelengths(self):
        pass

    @abstractmethod
    def select(self, srf: SpectralResponseFunction) -> SpectralGrid:
        """
        Select a subset of the spectral grid based on a reference data structure.
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
        raise NotImplementedError

    @select.register
    def _(self, srf: BandSRF) -> MonoSpectralGrid:
        raise NotImplementedError


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

    def select(self, srf: SpectralResponseFunction):
        # TODO: Support selection by uniform, delta and band SRFs
        raise NotImplementedError
