from __future__ import annotations

import logging
import typing as t

import attrs
import numpy as np
import pint
import pinttr
import xarray as xr

from .index import CKDSpectralIndex
from .spectral_set import SpectralSet
from ..attrs import documented, parse_docs
from ..constants import SPECTRAL_RANGE_MAX, SPECTRAL_RANGE_MIN
from ..quad import Quad, QuadType
from ..units import to_quantity
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg
from ..util.misc import round_to_multiple

logger = logging.getLogger(__name__)


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
        doc="Bin lower spectral bound.\n\nUnit-enabled field "
        '(default: ucc["wavelength"]).',
        type="quantity",
        init_type="quantity or float",
    )

    wmax: pint.Quantity = documented(
        pinttr.field(
            units=ucc.deferred("wavelength"),
            on_setattr=None,  # frozen instance: on_setattr must be disabled
        ),
        doc="Bin upper spectral bound.\n\nUnit-enabled field "
        '(default: ucc["wavelength"]).',
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

    @property
    def width(self) -> pint.Quantity:
        """quantity : Bin spectral width."""
        return self.wmax - self.wmin

    @property
    def wcenter(self) -> pint.Quantity:
        """quantity : Bin central wavelength."""
        return 0.5 * (self.wmin + self.wmax)

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
#                          CKD quadrature setup classes
# ------------------------------------------------------------------------------


@parse_docs
@attrs.define
class QuadSpec:
    """
    Abstract base class for all quadrature specification patterns.

    Each subclass defines a strategy used to generate a spectral quadrature
    corresponding to a CKD dataset and must implement the strategy in the
    :meth:`make_quad`.
    """

    @staticmethod
    def default() -> QuadSpecFixed:
        """
        Return the default spectral quadrature (Gauss-Legendre, 16 *g*-points).
        """
        return QuadSpecFixed(n=16, quad_type="gauss_legendre")

    @staticmethod
    def from_dict(
        value: dict[str, t.Any],
    ) -> QuadSpecFixed | QuadSpecMinError | QuadSpecErrorThreshold:
        """
        Create a quadrature specification subtype from a dictionary. The
        dictionary must have a ``type`` entry, whose value maps to a give
        quadrature specification subtype as follows:

        * ``fixed``: :class:`.QuadSpecFixed`
        * ``minimize_error``: :class:`.QuadSpecMinError`
        * ``error_threshold``: :class:`.QuadSpecErrorThreshold`

        Parameters
        ----------
        value : dict
            A dictionary mapping parameter names to their respective values.

        Returns
        -------
        QuadSpec
        """
        try:
            subtype: str = value.pop("type")
        except KeyError:
            raise ValueError("dictionary input must have a 'type' entry")

        if subtype == "fixed":
            cls = QuadSpecFixed
        elif subtype in {"minimize", "minimize_error"}:
            cls = QuadSpecMinError
        elif subtype in {"threshold", "error_threshold"}:
            cls = QuadSpecErrorThreshold
        else:
            raise ValueError(f"Unknown quadrature specification '{subtype}'")

        return cls.from_dict(value)

    @classmethod
    def convert(cls, value: t.Any) -> QuadSpec:
        """
        Attempt conversion to a :class:`.QuadSpec` instance. If `value` is a
        dictionary, it is passed to :meth:`.from_dict`; otherwise, it is left
        unchanged.
        """
        if isinstance(value, dict):
            return cls.from_dict(value)
        else:
            return value

    def make_quad(self, dataset: xr.Dataset) -> Quad:
        """
        Apply the quadrature generation strategy and generate a quadrature rule
        for a given dataset.

        Parameters
        ----------
        dataset : Dataset
            An xarray dataset following the CKD absorption data format, for
            which a quadrature rule is generated.

        Returns
        -------
        .Quad
        """
        raise NotImplementedError


@parse_docs
@attrs.define
class QuadSpecFixed(QuadSpec):
    """
    Fixed number of quadrature points [``fixed``]

    Use a fixed number of quadrature points for all bins. If the quadrature
    is specified this way, the quadrature type has to be explicitly specified
    using the ``type`` field.
    """

    n: int = documented(
        attrs.field(),
        doc="Number of quadrature points",
        type="int",
    )

    quad_type: QuadType = documented(
        attrs.field(default="gauss_legendre", converter=QuadType),
        doc="Quadrature type",
        type=".QuadType",
        init_type=".QuadType or str",
        default='"gauss_legendre"',
    )

    @classmethod
    def from_dict(cls, value: dict[str, t.Any]) -> QuadSpecFixed:
        return cls(**value)

    def make_quad(self, dataset: xr.Dataset) -> Quad:
        # Inherit docstring
        return Quad.new(type=self.quad_type, n=self.n)


@parse_docs
@attrs.define
class QuadSpecMinError(QuadSpec):
    """
    Error-minimizing number of quadrature points [``minimize_error``]

    Find the number of quadrature points that minimizes the error on the
    atmospheric transmittance. The quadrature type
    """

    nmax: int | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(int),
        ),
        doc="Maximum number of quadrature points",
        type="int or None",
        init_type="int, optional",
    )

    @classmethod
    def from_dict(cls, value: dict[str, t.Any]) -> QuadSpecMinError:
        return cls(**value)

    def make_quad(self, dataset: xr.Dataset) -> Quad:
        # Inherit docstring
        n = ng_minimum(error=dataset.error, ng_max=self.nmax)
        quad_type = dataset.ng.attrs.get("quadrature_type", "gauss_legendre")
        return Quad.new(type=quad_type, n=n)


@parse_docs
@attrs.define
class QuadSpecErrorThreshold(QuadSpec):
    """
    Error-threshold number of quadrature points [``error_threshold``]

    Find the number of quadrature points so that the error on the atmospheric
    transmittance is below a specified threshold.
    """

    threshold: float = documented(
        attrs.field(),
        doc="Error threshold value",
        type="float",
    )

    nmax: int | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(int),
        ),
        doc="Maximum number of quadrature points",
        type="int or None",
        init_type="int, optional",
    )

    @classmethod
    def from_dict(cls, value: dict[str, t.Any]) -> QuadSpecErrorThreshold:
        return cls(**value)

    def make_quad(self, dataset: xr.Dataset) -> Quad:
        # Inherit docstring
        quad_type = dataset.ng.attrs.get("quadrature_type", "gauss_legendre")
        n = ng_threshold(
            error=dataset.error, threshold=self.threshold, ng_max=self.nmax
        )
        return Quad.new(type=quad_type, n=n)


def ng_minimum(error: xr.DataArray, ng_max: int | None = None):
    """
    Find the number of quadrature points that minimizes the error.

    Parameters
    ----------
    error : DataArray
        Error data.

    ng_max : int, optional
        Maximum number of quadrature points. If not provided, it will be
        inferred from the error data.

    Returns
    -------
    int
        Number of quadrature points that minimizes the error.
    """

    if ng_max is None:
        ng_max = int(error.ng.max())

    error_w0 = error.isel(w=0)
    ng_min = int(error.ng.where(error_w0 == error_w0.min(), drop=True)[0])
    return ng_max if ng_min > ng_max else ng_min


def ng_threshold(
    error: xr.DataArray,
    threshold: float,
    ng_max: int | None = None,
):
    """
    Find the number of quadrature points so that the error is (strictly) below
    a specified threshold value.

    Parameters
    ----------
    error : DataArray
        Error data.

    threshold : float
        Error threshold.

    ng_max : int, optional
        Maximum number of quadrature points. If not provided, it will be
        inferred from the error data.

    Returns
    -------
    int
        Number of quadrature points so that the error is below the threshold.
    """

    if ng_max is None:
        ng_max = int(error.ng.max())

    error_w0 = error.isel(w=0)
    ng = error.ng.where(error_w0 < threshold, drop=True)

    if ng.size == 0:
        return ng_max
    else:
        ng = int(ng[0])
        return ng_max if ng > ng_max else ng


# ------------------------------------------------------------------------------
#                              Bin set data class
# ------------------------------------------------------------------------------


@parse_docs
@attrs.define(eq=False, frozen=True, slots=True)
class BinSet(SpectralSet):
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
        type="list of :class:`.Bin`",
        init_type="iterable of :class:`.Bin`",
    )

    def spectral_indices(self) -> t.Generator[CKDSpectralIndex]:
        for bin in self.bins:
            yield from bin.spectral_indices()

    @property
    def wavelengths(self) -> pint.Quantity:
        return self.wcenters

    @property
    def wcenters(self) -> pint.Quantity:
        """
        Return the central wavelength of all bins.
        """
        units = ucc.get("wavelength")
        return [bin.wcenter.m_as(units) for bin in self.bins] * units

    @property
    def wmins(self) -> pint.Quantity:
        """
        Return the lower bound of all bins.
        """
        units = ucc.get("wavelength")
        return [bin.wmin.m_as(units) for bin in self.bins] * units

    @property
    def wmaxs(self) -> pint.Quantity:
        """
        Return the upper bound of all bins.
        """
        units = ucc.get("wavelength")
        return [bin.wmax.m_as(units) for bin in self.bins] * units

    @classmethod
    @ureg.wraps(None, (None, "nm", "nm", "nm", None), strict=False)
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
        start : quantity or float
            Lower bound of first bin. If a float is passed, it is interpreted as
            being in units of nm.

        stop : quantity
            Upper bound of last bin. If a float is passed, it is interpreted as
            being in units of nm.

        step : quantity, default: 10 nm
            Bin width. If a float is passed, it is interpreted as being in units
            of nm.

        quad : .Quad, optional
            Quadrature rule (same for all bins in the set). Defaults to
            a one-point Gauss-Legendre quadrature.

        Returns
        -------
        :class:`.BinSet`
            Generated bin set.
        """
        if quad is None:
            quad = Quad.gauss_legendre(1)

        wmins = np.arange(start, stop, step)
        wmaxs = wmins + step

        wunits = ucc.get("wavelength")
        bins = [
            Bin(
                wmin=(wmin * ureg.nm).to(wunits),
                wmax=(wmax * ureg.nm).to(wunits),
                quad=quad,
            )
            for wmin, wmax in zip(wmins, wmaxs)
        ]

        return cls(bins)

    @classmethod
    def from_srf(
        cls,
        srf: xr.Dataset,
        step: pint.Quantity = 10.0 * ureg.nm,
        quad: Quad | None = None,
    ) -> BinSet:
        """
        Generate a bin set with linearly spaced bins covering the spectral
        range of a spectral response function.

        Parameters
        ----------
        srf : Dataset
            Spectral response function dataset.

        step : quantity
            Wavelength step.

        quad : .Quad, optional
            Quadrature rule (same for all bins in the set). Defaults to
            a one-point Gauss-Legendre quadrature.

        Returns
        -------
        :class:`.BinSet`
            Generated bin set.
        """
        wavelengths = to_quantity(srf.w)
        wmin = wavelengths.min()
        wmax = wavelengths.max()

        return cls.arange(start=wmin - step, stop=wmax + step, step=step, quad=quad)

    @classmethod
    def from_wavelength_bounds(
        cls, wmin: pint.Quantity, wmax: pint.Quantity, quad: Quad | None = None
    ) -> BinSet:
        if quad is None:
            quad = Quad.gauss_legendre(1)

        return cls(
            bins=[
                Bin(wmin=_wmin, wmax=_wmax, quad=quad)
                for _wmin, _wmax in zip(np.atleast_1d(wmin), np.atleast_1d(wmax))
            ]
        )

    @classmethod
    def from_absorption_data(
        cls,
        datasets: xr.Dataset | t.Sequence[xr.Dataset],
        quad_spec: QuadSpec | None = None,
    ) -> BinSet:
        """
        Generate a bin set from one or several absorption datasets.

        Parameters
        ----------
        datasets : Dataset or sequence of Dataset
            Absorption dataset.

        quad_spec : .QuadSpec
            Quadrature rule specification. If provided, it will be used to
            generate the quadrature rule based on error data in the
            absorption dataset.

        Returns
        -------
        :class:`.BinSet`
            Generated bin set.

        Notes
        -----
        Assumes that the absorption datasets have a ``wbounds`` data variable.
        """
        if isinstance(datasets, xr.Dataset):
            datasets = [datasets]

        if quad_spec is None:
            quad_spec = QuadSpec.default()

        bins = []

        for dataset in datasets:
            # make quadrature rule
            quad = quad_spec.make_quad(dataset)

            # determine wavelength bounds
            wlower = to_quantity(dataset.wbounds.sel(wbv="lower"))
            wupper = to_quantity(dataset.wbounds.sel(wbv="upper"))

            if wlower.check("[length]"):
                wmin = wlower
                wmax = wupper
            elif wlower.check("[length]^-1"):
                wmin = (1.0 / wupper).to("nm")  # min wavelength is max wavenumber
                wmax = (1.0 / wlower).to("nm")  # max wavelength is min wavenumber
            else:
                raise ValueError(
                    f"Invalid dimensionality for dataset spectral coordinate; "
                    f"expected [length] or [length]^-1 "
                    f"(got {wlower.dimensionality})"
                )
            binset = cls.from_wavelength_bounds(wmin=wmin, wmax=wmax, quad=quad)
            bins.extend(binset.bins)

        return cls(bins=bins)

    @classmethod
    def default(cls):
        """
        Generate a default bin set, which covers Eradiate's default spectral
        range with 10 nm-wide bins.
        """
        wmin = round_to_multiple(SPECTRAL_RANGE_MIN.m_as(ureg.nm), 10.0, "nearest")
        wmax = round_to_multiple(SPECTRAL_RANGE_MAX.m_as(ureg.nm), 10.0, "nearest")
        dw = 10.0

        return BinSet.arange(
            start=wmin * ureg.nm, stop=wmax * ureg.nm, step=dw * ureg.nm
        )
