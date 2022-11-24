from __future__ import annotations

import typing as t

import attrs
import pint
import pinttr

from .attrs import documented, parse_docs
from .quad import Quad
from .units import unit_context_config as ucc

# ------------------------------------------------------------------------------
#                              CKD bin data classes
# ------------------------------------------------------------------------------


@parse_docs
@attrs.frozen
class Bin:
    """
    A data class representing a spectral bin in CKD modes.
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
            repr=lambda x: x.str_summary, validator=attrs.validators.instance_of(Quad)
        ),
        doc="Quadrature rule attached to the CKD bin.",
        type=":class:`.Quad`",
    )

    @property
    def width(self) -> pint.Quantity:
        """quantity : Bin width."""
        return self.wmax - self.wmin

    @property
    def w(self) -> pint.Quantity:
        """quantity : Bin central wavelength."""
        return 0.5 * (self.wmin + self.wmax)

    @property
    def bings(self) -> t.List[Bing]:
        """list of :class:`.Bing` : List of associated bings."""
        return [Bing(bin=self, g=g) for _, g in enumerate(self.quad.nodes)]

    @classmethod
    def convert(cls, value: t.Any) -> t.Any:
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
@attrs.frozen
class Bing:
    """
    A data class representing a CKD (bin, g) pair.
    """

    bin: Bin = documented(
        attrs.field(converter=Bin.convert),
        doc="CKD bin.",
        type=":class:`.Bin`",
    )

    g: float = documented(
        attrs.field(),
        doc="CKD g-point.",
        type="float",
    )

    @classmethod
    def convert(cls, value) -> t.Any:
        """
        If ``value`` is a tuple or a dictionary, try to construct a
        :class:`.Bing` instance from it. Otherwise, return ``value``
        unchanged.
        """
        if isinstance(value, tuple):
            return cls(*value)

        if isinstance(value, dict):
            return cls(**value)

        return value