from __future__ import annotations

import logging
import typing as t

import attrs
import numpy as np
import pint
import pinttr
import portion as P
import xarray as xr

from .index import CKDSpectralIndex
from ..attrs import documented, parse_docs
from ..constants import SPECTRAL_RANGE_MAX, SPECTRAL_RANGE_MIN
from ..quad import Quad
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg
from ..util.misc import round_to_multiple

logger = logging.getLogger(__name__)

# G16 = Quad.gauss_legendre(16).eval_nodes(interval=[0.0, 1.0])  # TODO: PR#311 hack
# Note: Hard-coded because Mitsuba cannot be imported at module level due to
#       documentation issues.
G16 = np.array(
    [
        0.00529954,
        0.02771249,
        0.06718439,
        0.12229779,
        0.19106188,
        0.27099161,
        0.35919823,
        0.45249375,
        0.54750625,
        0.64080177,
        0.72900839,
        0.80893812,
        0.87770221,
        0.93281561,
        0.97228751,
        0.99470046,
    ]
)


# ------------------------------------------------------------------------------
#                              CKD bin data classes
# ------------------------------------------------------------------------------


@parse_docs
@attrs.define(eq=False, frozen=True, slots=True)
class Bin:
    """
    A data class representing a spectral bin in CKD modes.

    Notes
    -----
    A bin is more than a spectral interval. It is associated with a
    quadrature rule.
    """

    wmin: pint.Quantity = documented(
        pinttr.field(
            units=ucc.deferred("wavelength"),
            on_setattr=None,  # frozen instance: on_setattr must be disabled
        ),
        doc='Bin lower spectral bound.\n\nUnit-enabled field (default: ucc["wavelength"]).',
        type="quantity",
        init_type="quantity or float",
    )

    wmax: pint.Quantity = documented(
        pinttr.field(
            units=ucc.deferred("wavelength"),
            on_setattr=None,  # frozen instance: on_setattr must be disabled
        ),
        doc='Bin upper spectral bound.\n\nUnit-enabled field (default: ucc["wavelength"]).',
        type="quantity",
        init_type="quantity or float",
    )

    @wmin.validator
    @wmax.validator
    def _wbounds_validator(self, attribute, value):
        if not self.wmin < self.wmax:
            raise ValueError(
                f"while validating {attribute.name}: wmin must be lower than wmax"
            )

    quad: Quad = documented(
        attrs.field(
            factory=lambda: Quad.gauss_legendre(2),
            repr=lambda x: x.str_summary,
            validator=attrs.validators.instance_of(Quad),
        ),
        doc="Quadrature rule attached to the CKD bin.",
        type=":class:`.Quad`",
    )

    bin_set_id: str | None = documented(
        attrs.field(default=None, converter=str),
        doc="Id of the bin set used to create this bin.",
        type="str or None",
        init_type="str",
    )

    @property
    def width(self) -> pint.Quantity:
        """quantity : Bin spectral width."""
        return self.wmax - self.wmin

    @property
    def wcenter(self) -> pint.Quantity:
        """quantity : Bin central wavelength."""
        return 0.5 * (self.wmin + self.wmax)

    @property
    def interval(self) -> P.Interval:
        """portion.Interval : Closed-open interval corresponding to the bin
        wavelength interval."""
        return P.closedopen(self.wmin, self.wmax)

    @property
    def pretty_repr(self) -> str:
        """str : Pretty representation of the bin."""
        units = ureg.Unit("nm")
        wrange = (
            f"[{self.wmin.m_as(units):.1f}, {self.wmax.m_as(units):.1f}] {units:~P}"
        )
        quad = self.quad.pretty_repr()
        return f"{wrange}: {quad}"

    def spectral_indices(self) -> t.Generator[CKDSpectralIndex]:
        for value in self.quad.eval_nodes(interval=[0.0, 1.0]):
            yield CKDSpectralIndex(w=self.wcenter, g=value)


# ------------------------------------------------------------------------------
#                              Bin set data class
# ------------------------------------------------------------------------------


@parse_docs
@attrs.define(eq=False, frozen=True, slots=True)
class BinSet:
    """
    A data class representing a bin set used in CKD mode.

    See Also
    --------
    :class:`.WavelengthSet`
    """

    bins: list[Bin] = documented(
        attrs.field(
            converter=list,
            validator=attrs.validators.deep_iterable(
                member_validator=attrs.validators.instance_of(Bin)
            ),
        ),
        doc="Set of bins.",
        type="set of :class:`.Bin`",
        init_type="iterable of :class:`.Bin`",
    )

    def spectral_indices(self) -> t.Generator[CKDSpectralIndex]:
        for bin in self.bins:
            yield from bin.spectral_indices()

    @classmethod
    def arange(
        cls,
        start: pint.Quantity,
        stop: pint.Quantity,
        step: pint.Quantity = 10.0 * ureg.nm,
        quad: Quad | None = None,
    ) -> BinSet:
        """
        Generate a bin set with linearly spaced bins.

        Parameters
        ----------
        start : quantity
            Lower bound of first bin.

        stop : quantity
            Upper bound of last bin.

        step : quantity
            Bin width.

        quad : .Quad, optional
            Quadrature rule (same for all bins in the set). Defaults to
            a two-point Gauss-Legendre quadrature.

        Returns
        -------
        :class:`.BinSet`
            Generated bin set.
        """
        if quad is None:
            quad = Quad.gauss_legendre(2)

        bins = []

        wunits = ucc.get("wavelength")
        _start = start.m_as(wunits)
        _stop = stop.m_as(wunits)
        _step = step.m_as(wunits)

        for wmin in np.arange(_start, _stop, _step):
            wmax = wmin + _step
            bins.append(
                Bin(
                    wmin=wmin * wunits,
                    wmax=wmax * wunits,
                    quad=quad,
                )
            )

        return cls(bins)

    @classmethod
    def from_wavelengths(
        cls,
        wavelengths: pint.Quantity,
        width: pint.Quantity = 10.0 * ureg.nm,
        quad: Quad | None = None,
    ) -> BinSet:
        """
        Generate a bin set with bins centered on the given wavelengths.

        Parameters
        ----------
        wavelengths : sequence of quantity
            Wavelengths to center bins on.

        quad : :class:`.Quad`, optional
            Quadrature rule (same for all bins in the set). Defaults to
            a two-point Gauss-Legendre quadrature.

        Returns
        -------
        :class:`.BinSet`
            Generated bin set.
        """
        if quad is None:
            quad = Quad.gauss_legendre(2)

        bins = []

        for wcenter in np.atleast_1d(wavelengths):
            bins.append(
                Bin(
                    wmin=wcenter - width / 2,
                    wmax=wcenter + width / 2,
                    quad=quad,
                )
            )

        return cls(bins)

    @classmethod
    def from_absorption_dataset(
        cls,
        dataset: xr.Dataset,
        quad: Quad | None = None,
    ) -> BinSet:
        """
        Generate a bin set from an absorption dataset.

        Parameters
        ----------
        dataset : Dataset
            Absorption dataset.

        quad : :class:`.Quad`, optional
            Quadrature rule (same for all bins in the set). Defaults to
            an eight-point Gauss-Legendre quadrature.

        Returns
        -------
        :class:`.BinSet`
            Generated bin set.

        Notes
        -----
        Assumes that:

        * the absorption dataset has a ``bin`` coordinate with values in
          nanometers.
        * the absorption dataset has a ``bin_set`` attribute with the bin
          width in nanometers.
        """
        if quad is None:
            quad = Quad.gauss_legendre(8)

        # TODO: PR#311 hack (next 2 lines)
        wavelengths = np.unique(np.array(dataset.bin.values, dtype=float)) * ureg.nm
        width = ureg(
            dataset.attrs["bin_set"]
        )  # read bin_set dataset attribute and convert it to a quantity
        return cls.from_wavelengths(wavelengths, width=width, quad=quad)

    @classmethod
    def default(cls):
        """
        Generate a default bin set, which covers Eradiate's default spectral
        range with 10 nm-wide bins.
        """
        wmin = round_to_multiple(SPECTRAL_RANGE_MIN.m_as(ureg.nm), 10.0, "nearest")
        wmax = round_to_multiple(SPECTRAL_RANGE_MAX.m_as(ureg.nm), 10.0, "nearest")
        dw = 10.0

        return cls.from_wavelengths(
            wavelengths=np.arange(wmin, wmax + dw, dw) * ureg.nm,
            width=dw * ureg.nm,
            quad=Quad.gauss_legendre(2),
        )
