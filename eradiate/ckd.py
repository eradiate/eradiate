from __future__ import annotations

import functools
from collections import abc
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import attr
import pint
import pinttr
import xarray as xr

from . import data
from .attrs import documented, parse_docs
from .quad import Quad
from .units import unit_context_config as ucc
from .units import unit_registry as ureg

# ------------------------------------------------------------------------------
#                              CKD bin data classes
# ------------------------------------------------------------------------------


@parse_docs
@attr.s(frozen=True)
class Bin:
    """
    A data class representing a spectral bin in CKD modes.
    """

    id: str = documented(
        attr.ib(converter=str),
        doc="Bin identifier.",
        type="str",
    )

    wmin: pint.Quantity = documented(
        pinttr.ib(
            units=ucc.deferred("wavelength"),
            on_setattr=None,  # frozen instance: on_setattr must be disabled
        ),
        doc='Bin lower spectral bound.\n\nUnit-enabled field (default: ucc["wavelength"]).',
        type="float",
    )

    wmax: pint.Quantity = documented(
        pinttr.ib(
            units=ucc.deferred("wavelength"),
            on_setattr=None,  # frozen instance: on_setattr must be disabled
        ),
        doc='Bin upper spectral bound.\n\nUnit-enabled field (default: ucc["wavelength"]).',
        type="float",
    )

    @wmin.validator
    @wmax.validator
    def _wbounds_validator(self, attribute, value):
        if not self.wmin < self.wmax:
            raise ValueError(
                f"while validating {attribute.name}: wmin must be lower than wmax"
            )

    quad: Quad = documented(
        attr.ib(
            repr=lambda x: x.str_summary, validator=attr.validators.instance_of(Quad)
        ),
        doc="Quadrature rule attached to the CKD bin.",
        type=":class:`.Quad`",
    )

    @property
    def width(self) -> pint.Quantity:
        """Bin spectral width."""
        return self.wmax - self.wmin

    @property
    def wcenter(self) -> pint.Quantity:
        """Bin central wavelength."""
        return 0.5 * (self.wmin + self.wmax)

    @property
    def bindexes(self) -> List[Bindex]:
        return [Bindex(bin=self, index=i) for i, _ in enumerate(self.quad.nodes)]

    @classmethod
    def convert(cls, value):
        """
        If ``value`` is a tuple or a dictionary, try to construct a
        :class:`.Bin` instance from it. Otherwise, return ``value`` unchanged.
        """
        if isinstance(value, tuple):
            return cls(*value)

        if isinstance(value, dict):
            return cls(**value)

        return value


@parse_docs
@attr.s(frozen=True)
class Bindex:
    """
    A data class representing a CKD (bin, index) pair.
    """

    bin: Bin = documented(
        attr.ib(converter=Bin.convert),
        doc="CKD bin.",
        type=":class:`.Bin`",
    )

    index: int = documented(
        attr.ib(int),
        doc="Quadrature point index.",
        type="int",
    )

    @classmethod
    def convert(cls, value) -> Any:
        """
        If ``value`` is a tuple or a dictionary, try to construct a
        :class:`.Bindex` instance from it. Otherwise, return ``value``
        unchanged.
        """
        if isinstance(value, tuple):
            return cls(*value)

        if isinstance(value, dict):
            return cls(**value)

        return value


# ------------------------------------------------------------------------------
#                           Bin filtering functions
# ------------------------------------------------------------------------------

# The following functions implement a set of bin selection routines used to
# filter and select bins from a bin set.


def bin_filter_ids(ids: Sequence[str]) -> Callable[[Bin], bool]:
    """
    Select bins based on identifiers.

    Parameter ``ids`` (list[str]):
        A sequence of bin identifiers which the defined filter will let through.

    Returns → callable:
        A callable which returns ``True`` *iff* a bin has a valid identifier.
    """

    def filter(value):
        return value.id in ids

    return filter


def bin_filter_interval(
    wmin: pint.Quantity, wmax: pint.Quantity, endpoints: bool = True
) -> Callable[[Bin], bool]:
    """
    Select bins in a wavelength interval.

    Parameter ``wmin`` (:class:`pint.Quantity`):
        Lower bound of the spectral interval defining the filter.

    Parameter ``wmax`` (:class:`pint.Quantity`):
        Upper bound of the spectral interval defining the filter. Must be
        equal to or greater than ``wmin``.

    Parameter ``endpoints`` (bool):
        If ``True``, bins must have at least one of their bounds within the
        interval to be selected.
        If ``False``, bins must have both bounds in the interval to be
        selected.

    Returns → callable:
        A callable which returns ``True`` *iff* a bin is in the specified interval.

    Raises → ValueError:
        If ``wmin > wmax``.
    """
    if wmin > wmax:
        raise ValueError("wmin must be lower or equal to wmax")

    if wmin == wmax:

        def filter(value):
            return value.wmin < wmin < value.wmax

    else:
        if endpoints:

            def filter(value):
                return (
                    (wmin <= value.wmin and value.wmax <= wmax)
                    or (value.wmin < wmin < value.wmax)
                    or (value.wmin < wmax < value.wmax)
                )

        else:

            def filter(value):
                return wmin <= value.wmin and value.wmax <= wmax

    return filter


def bin_filter(type: str, filter_kwargs: Dict[str, Any]):
    """
    Create a bin filter function dynamically.

    Parameter ``type`` (str):
        Filter type.

    Parameter ``filter_kwargs`` (dict):
        Keyword arguments passed to the filter generator.

    Returns → callable:
        Generated bin filter function.

    .. note::
       Valid filter types are:

       * ``all`` (``lambda x: True``);
       * ``ids`` (:func:`.bin_filter_ids`);
       * ``interval`` (:func:`.bin_filter_interval`).

       See the corresponding API entry for expected arguments.
    """
    if type == "interval":
        return bin_filter_interval(**filter_kwargs)

    if type == "ids":
        return bin_filter_ids(**filter_kwargs)

    if type == "all":
        return lambda x: True

    else:
        raise ValueError(f"unknown bin filter type {type}")


# ------------------------------------------------------------------------------
#                              Bin set definitions
# ------------------------------------------------------------------------------


@parse_docs
@attr.s(frozen=True)
class BinSet:
    """
    A data class representing a quadrature definition used in CKD mode.
    """

    id: str = documented(
        attr.ib(converter=str),
        doc="Bin set identifier.",
        type="str",
    )

    quad: Quad = documented(
        attr.ib(repr=lambda x: x.str_summary),
        doc="Quadrature rule associated with this spectral bin set.",
        type=":class:`.Quad`",
    )

    bins: Tuple[Bin, ...] = documented(
        attr.ib(
            converter=lambda x: tuple(
                sorted(
                    x, key=lambda y: (y.wmin, y.wmax, y.id)
                )  # Specify field priority for sorting and exclude quad (which is irrelevant)
            ),
            validator=attr.validators.deep_iterable(
                member_validator=attr.validators.instance_of(Bin)
            ),
            repr=lambda x: f"tuple<{len(x)}>("
            f"Bin({repr(x[0].id)}, ... ), "
            f"Bin({repr(x[1].id)}, ... ), ... , "
            f"Bin({repr(x[-2].id)}, ... ), "
            f"Bin({repr(x[-1].id)}, ... ))"
            if len(x) >= 5
            else f"{tuple(y.id for y in x)}",
        ),
        doc="List of bins (automatically ordered upon setting).",
        type="tuple[:class:`.Bin`, ...]",
    )

    @bins.validator
    def _bins_validator(self, attribute, value):
        if any(bin.quad is not self.quad for bin in value):
            raise ValueError(
                f"while validating {attribute.name}: all defined bins must "
                "share the same quadrature as their parent bin set"
            )

    @property
    def bin_ids(self) -> List[str]:
        return [bin.id for bin in self.bins]

    @property
    def bin_wmins(self) -> pint.Quantity:
        units = ucc.get("wavelength")
        return ureg.Quantity([bin.wmin.m_as(units) for bin in self.bins], units)

    @property
    def bin_wmaxs(self) -> pint.Quantity:
        units = ucc.get("wavelength")
        return ureg.Quantity([bin.wmax.m_as(units) for bin in self.bins], units)

    def filter_bins(self, *filters: Callable[[Bin], bool]) -> Tuple[Bin, ...]:
        """
        Filter bins based on callables.

        Parameter ``filters`` (callable):
            One or several callables with signature ``filter(x: Bin) -> bool``.

        Returns → dict[str, :class:`.Bin`]:
            Only bins for which ``filter(bin)`` is ``True`` are returned,
            ordered by their lower bound.
        """
        selected_bins = []

        for bin in self.bins:
            for filter in filters:
                if filter(bin):
                    selected_bins.append(bin)
                    break

        return tuple(sorted(selected_bins, key=lambda x: (x.wmin, x.wmax, x.id)))

    def select_bins(
        self,
        *filter_specs: Union[
            str,
            Callable[[Bin], bool],
            Tuple[str, Dict[str, Any]],
            Dict,
        ],
    ) -> Tuple[Bin, ...]:
        """
        Select a subset of CKD bins. This method is a high-level wrapper for
        :meth:`.filter_bins`.

        Parameter ``filter_specs`` (sequence[str or callable or sequence or dict]):
            One or several bin filter specifications. The following are supported:

            * a string will select a bin with matching ID (internally, this
              results in a call to :func:`.bin_filter_ids`);
            * a callable will be directly used;
            * a (str, dict) will be interpreted as the ``type`` and
              ``filter_kwargs`` arguments of :func:`.bin_filter`;
            * a dict with keys ``type`` and ``filter_kwargs`` will be directly
              forwarded as keyword arguments to :func:`.bin_filter`.

        Returns → dict[str, :class:`.Bin`]:
            Selected bins.
        """
        filters = []

        for filter_spec in filter_specs:
            if isinstance(filter_spec, str):
                filters.append(bin_filter_ids(ids=(filter_spec,)))

            elif isinstance(filter_spec, Sequence):
                filters.append(bin_filter(*filter_spec))

            elif isinstance(filter_spec, abc.Mapping):
                filters.append(bin_filter(**filter_spec))

            elif callable(filter_spec):
                filters.append(filter_spec)

            else:
                raise ValueError(f"unhandled CKD bin selector {filter_spec}")

        return self.filter_bins(*filters)

    @classmethod
    def convert(cls, value) -> Any:
        """
        If ``value`` is a string, query quadrature definition database with it.
        Otherwise, return ``value`` unchanged.
        """
        if isinstance(value, str):
            return cls.from_db(value)

        return value

    @staticmethod
    def from_dataset(id: str, ds: xr.Dataset) -> BinSet:
        """
        Convert a dataset-based bin set definition to a :class:`BinSet`
        instance.

        Parameter ``id`` (str):
            Data set identifier.

        Parameter ``ds`` (:class:`~xarray.Dataset`):
            Dataset from which bin set definition information is to be
            extracted.

        Returns → :class:`.BinSet`:
            Bin set definition.
        """
        # Collect quadrature data
        quad_type = ds.attrs["quadrature_type"]
        quad_n = ds.attrs["quadrature_n"]
        quad = Quad.new(quad_type, quad_n)

        # Collect bin set data
        bin_ids = ds.bin.values
        bin_wmin = ureg.Quantity(ds.wmin.values, ds.wmin.attrs["units"])
        bin_wmax = ureg.Quantity(ds.wmax.values, ds.wmax.attrs["units"])

        # Assemble the data
        return BinSet(
            id=id,
            quad=quad,
            bins=tuple(
                Bin(id=id, wmin=wmin, wmax=wmax, quad=quad)
                for id, wmin, wmax in zip(bin_ids, bin_wmin, bin_wmax)
            ),
        )

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def from_db(id: str) -> BinSet:
        """
        Get a bin set definition from Eradiate's database.

        .. note::
           This static function is cached using :func:`functools.lru_cache` for
           optimal performance.

        Parameter ``id`` (str):
            Data set identifier. The :func:`eradiate.data.open` function will be
            used to load the requested data set.

        Returns → :class:`.BinSet`:
            Bin set definition.
        """
        ds = data.open("ckd_bin_set", id)
        return BinSet.from_dataset(id=id, ds=ds)

    @staticmethod
    def from_node_dataset(ds: xr.Dataset):
        """
        Get bin set from node data.

        Parameter ``ds`` (:class:`~xarray.Dataset`):
            Node data to get the bin set for. Data set attributes must have a
            ``bin_set_id`` field referencing a registered bin set definition in
            the Eradiate database.

        Returns → :class:`.BinSet`:
            Bin set definition associated with the passed CKD node data.
        """

        return BinSet.from_db(ds.attrs["bin_set"])
