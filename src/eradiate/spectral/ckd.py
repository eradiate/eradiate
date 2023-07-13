from __future__ import annotations

import itertools
import logging
import typing as t
from functools import singledispatch

import attrs
import numpy as np
import pint
import pinttr
import portion as P
import xarray as xr

from .index import CKDSpectralIndex
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
@attrs.define
class QuadratureSpecifications:
    r"""
    Quadrature rule specifications.

    Notes
    -----

    .. list-table:: Quadrature specifications types
       :widths: 30 30 30
       :header-rows: 1

       * - ``type``
         - Description
         - Parameters
       * - ``"fixed"``
         - Use a fixed number of quadrature points.
         - ``n`` (``int``): number of quadrature points,
           ``type`` (``str``): quadrature type (default: ``"gauss_legendre"``)
       * - ``"minimize_error"``
         - Find the number of quadrature points that minimizes the error on the
           atmospheric transmittance.
         - ``nmax`` (``int``): upper bound on the number of quadrature points
       * - ``"error_below_threshold"``
         - Find the number of quadrature points so that the error on the
           atmospheric transmittance is below a specified threshold.
         - ``threshold`` (``float``): error threshold value,
           ``nmax`` (``int``): upper bound on the number of quadrature points.
    """

    type: str = documented(
        attrs.field(
            default="fixed",
            converter=str,
            validator=attrs.validators.in_(
                {"fixed", "minimize_error", "error_below_threshold"}
            ),
        ),
        doc="Algorithm to determine the number of quadrature points."
        'Must be in {"fixed", "minimize_error", "error_below_threshold"}.'
        "Each algorithm takes additional parameters (see ``params``).",
        default="fixed",
        type="str",
        init_type="str",
    )

    params: dict = documented(
        attrs.field(
            default={"type": "gauss_legendre", "n": 1},
            converter=dict,
            validator=attrs.validators.instance_of(dict),
        ),
        doc="Parameters to the algorithm used to determine the number of "
        "quadrature points."
        "Refer to the notes section for the parameters corresponding to "
        "each algorithm.",
        default="{'type': 'gauss_legendre', 'n': 1}",
        type="dict",
        init_type="dict",
    )

    @type.validator
    @params.validator
    def _params_validator(self, attribute, value):
        # if type is "fixed", params must be a dict with keys "type" and "n"
        # where 'type' is a str and 'n' an int
        # elif type is "minimize_error", params must be a dict with key "nmax"
        # where 'nmax' is an int
        # elif type is "threshold", params must be a dict with keys 'threshold'
        # and 'nmax' where 'threshold' is a float and 'nmax' is an int

        if self.type == "fixed":
            if self.params.keys() != {"type", "n"}:
                raise ValueError(
                    f"while validating {attribute.name}: "
                    f"params must be a dict with keys 'type' and 'n'"
                    f"(got {list(self.params.keys())})"
                )
            if not isinstance(self.params["type"], str):
                raise ValueError(
                    f"while validating {attribute.name}: "
                    f"params['type'] must be a str"
                    f"(got {type(self.params['type'])})"
                )
            if not self.params["type"] in [x.value for x in QuadType]:
                raise ValueError(
                    f"while validating {attribute.name}: "
                    f"params['type'] must be in {[x.value for x in QuadType]} "
                    f"(got {self.params['type']})"
                )
            if not isinstance(self.params["n"], int):
                raise ValueError(
                    f"while validating {attribute.name}: "
                    f"params['n'] must be an int"
                    f"(got {type(self.params['n'])})"
                )

        elif self.type == "minimize_error":
            if self.params.keys() != {"nmax"}:
                raise ValueError(
                    f"while validating {attribute.name}: "
                    f"params must be a dict with key 'nmax' "
                    f"(got {list(self.params.keys())})"
                )
            if not isinstance(self.params["nmax"], int):
                raise ValueError(
                    f"while validating {attribute.name}: "
                    f"params['nmax'] must be an int"
                    f"(got {type(self.params['nmax'])})"
                )

        elif self.type == "error_below_threshold":
            if self.params.keys() != {"threshold", "nmax"}:
                raise ValueError(
                    f"while validating {attribute.name}: "
                    f"params must be a dict with keys 'threshold' and 'nmax'"
                    f"(got {list(self.params.keys())})"
                )
            if not isinstance(self.params["threshold"], float):
                raise ValueError(
                    f"while validating {attribute.name}: "
                    f"params['threshold'] must be a float"
                    f"(got {type(self.params['threshold'])})"
                )
            if not isinstance(self.params["nmax"], int):
                raise ValueError(
                    f"while validating {attribute.name}: "
                    "params['nmax'] must be an int"
                    f"(got {type(self.params['nmax'])})"
                )

    @classmethod
    def convert(
        cls, value: QuadratureSpecifications | dict
    ) -> QuadratureSpecifications:
        if isinstance(value, dict):
            return cls.from_dict(value)
        elif isinstance(value, QuadratureSpecifications):
            return value
        else:
            raise TypeError(
                f"Unsupported type {type(value)}; "
                f"expected dict or QuadratureSpecifications"
            )

    @classmethod
    def from_dict(cls, value: dict) -> QuadratureSpecifications:
        return cls(**value)

    def make_quad(self, dataset: xr.Dataset) -> Quad:
        """
        Make a quadrature rule from the specifications and the dataset.
        """
        if self.type == "fixed":
            return Quad.new(**self.params)

        elif self.type == "minimize_error":
            n = ng_minimum(
                error=dataset.error,
                ng_max=self.params.get("nmax", None),
            )
            quad_type = dataset.ng.attrs.get(
                "quadrature_type",
                "gauss_legendre",
            )
            return Quad.new(type=quad_type, n=n)

        elif self.type == "error_below_threshold":
            n = ng_threshold(
                error=dataset.error,
                threshold=self.params["threshold"],
                ng_max=self.params.get("nmax", None),
            )
            quad_type = dataset.ng.attrs.get(
                "quadrature_type",
                "gauss_legendre",
            )
            return Quad.new(type=quad_type, n=n)

        else:
            raise NotImplementedError(f"Unsupported type {self.type}")


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
            a one-point Gauss-Legendre quadrature.

        Returns
        -------
        :class:`.BinSet`
            Generated bin set.
        """
        if quad is None:
            quad = Quad.gauss_legendre(1)

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
    def from_srf(
        cls,
        srf: xr.Dataset,
        step: pint.Quantity = 10.0 * ureg.nm,
        quad: Quad | None = None,
    ) -> BinSet:
        """
        Generate a bin set with linearly spaced bins, that covers the spectral
        range of a spectral response function.

        Parameters
        ----------
        srf: Dataset
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

        return cls.arange(
            start=wmin - step,
            stop=wmax + step,
            step=step,
            quad=quad,
        )

    @classmethod
    def from_wavelength_bounds(
        cls,
        wmin: pint.Quantity,
        wmax: pint.Quantity,
        quad: Quad | None = None,
    ) -> BinSet:
        quad = Quad.gauss_legendre(1) if quad is None else quad

        return cls(
            bins=[
                Bin(wmin=_wmin, wmax=_wmax, quad=quad)
                for _wmin, _wmax in zip(np.atleast_1d(wmin), np.atleast_1d(wmax))
            ]
        )

    @classmethod
    def from_absorption_dataset(
        cls,
        dataset: xr.Dataset,
        quad_spec: QuadratureSpecifications = QuadratureSpecifications(),
    ) -> BinSet:
        """
        Generate a bin set from an absorption dataset.

        Parameters
        ----------
        dataset : Dataset
            Absorption dataset.

        quad_spec : QuadratureSpecifications
            Quadrature rule specification. If provided, it will be used to
            generate the quadrature rule based on error data in the
            absorption dataset.

        Returns
        -------
        :class:`.BinSet`
            Generated bin set.

        Notes
        -----
        Assumes that the absorption dataset has a ``wbounds`` data variable.
        """

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

        return cls.from_wavelength_bounds(
            wmin=wmin,
            wmax=wmax,
            quad=quad,
        )

    @classmethod
    def from_absorption_datasets(
        cls,
        datasets: list[xr.Dataset],
        quad_spec: QuadratureSpecifications = QuadratureSpecifications(),
    ) -> BinSet:
        """
        Generate a bin set from a list of absorption datasets.

        Parameters
        ----------
        datasets : list of Dataset
            Absorption datasets.

        quad_spec : QuadratureSpecifications
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
        binsets = [
            cls.from_absorption_dataset(dataset, quad_spec=quad_spec)
            for dataset in datasets
        ]
        return cls(bins=itertools.chain.from_iterable([b.bins for b in binsets]))

    @classmethod
    def from_absorption_data(cls, absorption_data, quad_spec: dict) -> BinSet:
        return from_absorption_data_impl(absorption_data, quad_spec=quad_spec)

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
            start=wmin * ureg.nm,
            stop=wmax * ureg.nm,
            step=dw * ureg.nm,
        )


@singledispatch
def from_absorption_data_impl(
    absorption_data: xr.Dataset | list,
    quad_spec: QuadratureSpecifications,
) -> BinSet:
    raise NotImplementedError(f"Unsupported type {type(absorption_data)}")


@from_absorption_data_impl.register(xr.Dataset)
def _(absorption_data, quad_spec: QuadratureSpecifications) -> BinSet:
    return BinSet.from_absorption_dataset(
        dataset=absorption_data,
        quad_spec=quad_spec,
    )


@from_absorption_data_impl.register(list)
def _(absorption_data, quad_spec: QuadratureSpecifications) -> BinSet:
    return BinSet.from_absorption_datasets(
        datasets=absorption_data,
        quad_spec=quad_spec,
    )
