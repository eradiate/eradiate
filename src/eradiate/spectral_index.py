from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

import attrs
import pint
import pinttr

import eradiate

from . import validators
from .attrs import documented, parse_docs
from .exceptions import ModeError
from .units import unit_context_config as ucc
from .units import unit_registry as ureg

# If you want to add a new spectral mode, you need to:
# * add a new SpectralIndex subclass
# * add it to the SPECTRAL_INDEX_DISPATCH dictionary.


@attrs.define(frozen=True)
class SpectralIndex(ABC):
    """Abstract spectral index base class.
    
    See Also
    --------
    :class:`MonoSpectralIndex`, :class:`CKDSpectralIndex`
    """

    @property
    @abstractmethod
    def formatted_repr(self) -> str:
        """Return a formatted representation of the spectral index."""
        pass

    @property
    @abstractmethod
    def as_hashable(self) -> t.Hashable:
        """Return a hashable representation of the spectral index.
        
        Notes
        -----
        The hashable value is used by :class:`eradiate.experiments.Experiment`s
        to identify the simulation results corresponding to a given spectral
        index.
        """
        pass

    @staticmethod
    def new(mode: t.Optional[str] = None, **kwargs) -> SpectralIndex:
        """Spectral index factory method.
        
        Parameters
        ----------
        mode : str, optional
            Spectral mode identifier (``"mono"`` or ``"ckd"``).
            If ``None``, the current mode is used.
        
        Returns
        -------
        SpectralIndex
            Spectral index instance.
        
        Notes
        -----
        Create a new instance of the spectral index class corresponding to
        the specified mode.

        See Also
        --------
        :class:`MonoSpectralIndex`, :class:`CKDSpectralIndex`
        """
        if mode is None:
            if eradiate.mode().is_mono():
                mode = "mono"
            elif eradiate.mode().is_ckd():
                mode = "ckd"
            else:
                raise ModeError(
                    "Unsupported spectral mode: None. Must be one of "
                    f"{list(SPECTRAL_INDEX_DISPATCH.keys())}."
                )
        try:
            spectral_index_cls = SPECTRAL_INDEX_DISPATCH[mode]
            return spectral_index_cls(**kwargs)
        except KeyError as e:
            raise ModeError(
                f"Unsupported spectral mode: {mode}. Must be one of "        
                f"{list(SPECTRAL_INDEX_DISPATCH.keys())}."
            ) from e

    @staticmethod
    def from_dict(d: t.Dict[str, t.Any]) -> SpectralIndex:
        d_copy = pinttr.interpret_units(d, ureg=ureg)
        return SpectralIndex.new(**d_copy)


    @staticmethod
    def convert(value: t.Any) -> SpectralIndex:
        if isinstance(value, SpectralIndex):
            return value
        elif isinstance(value, dict):
            return SpectralIndex.from_dict(value)
        else:
            raise ValueError(f"Cannot convert {value} to a spectral index.")

@parse_docs
@attrs.define(frozen=True)
class MonoSpectralIndex(SpectralIndex):
    """Monochromatic spectral index.
    
    Parameters
    ----------
    wavelength : pint.Quantity
        Wavelength.

    Attributes
    ----------
    wavelength : pint.Quantity
        Wavelength.
    
    See Also
    --------
    :class:`CKDSpectralIndex`
    """
    w: pint.Quantity = documented(
        pinttr.field(
            default=550.0 * ureg.nm,
            units=ucc.deferred("length"),
            on_setattr=None,
        ),
        doc="Wavelength.\n\nUnit-enabled field (default: ucc[length]).",
        type="quantity",
        init_type="quantity or float",
        default="550.0 nm",
    )

    @w.validator
    def _w_validator(self, attribute, value):

        # wavelength must be a scalar quantity
        validators.on_quantity(validators.is_scalar)(self, attribute, value)

        # wavelength msut be positive
        validators.is_positive(self, attribute, value)

    @property
    def formatted_repr(self) -> str:
        return f"{self.w:g~P}"
    
    @property
    def as_hashable(self) -> float:
        return float(self.w.m_as(ureg.nm))
    

@parse_docs
@attrs.define(frozen=True)
class CKDSpectralIndex(SpectralIndex):
    """CKD spectral index.
    
    Parameters
    ----------
    w : pint.Quantity
        Bin center wavelength.
    
    g : float
        g value.

    Attributes
    ----------
    w : pint.Quantity
        Bin center wavelength.
    
    g : float
        g value.
    
    See Also
    --------
    :class:`MonoSpectralIndex`
    """
    w: pint.Quantity = documented(
        pinttr.field(
            default=550.0 * ureg.nm,
            units=ucc.deferred("length"),
            on_setattr=None,
        ),
        doc="Bin center wavelength.\n\nUnit-enabled field (default: ucc[length]).",
        type="quantity",
        init_type="quantity or float",
        default="550.0 nm",
    )

    @w.validator
    def _w_validator(self, attribute, value):
        
        # wavelength must be a scalar quantity
        validators.on_quantity(validators.is_scalar)(self, attribute, value)

        # wavelength magnitude must be positive
        validators.is_positive(self, attribute, value.magnitude)

    g: float = documented(
        attrs.field(
            default=0.0,
            validator=attrs.validators.instance_of(float),
            converter=float,
        ),
        doc="g value.",
        type="float",
        init_type="float",
        default="0.0",
    )

    @g.validator
    def _g_validator(self, attribute, value):
        # g value must be between 0 and 1
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{attribute} must be between 0 and 1, got {value}")

    @property
    def formatted_repr(self) -> str:
        return f"{self.w:g~P}:{self.g:g}"
    
    @property
    def as_hashable(self) -> t.Tuple[float, float]:
        return (float(self.w.m_as(ureg.nm)), self.g)


SPECTRAL_INDEX_DISPATCH: t.Dict[str, SpectralIndex] = {
    "mono": MonoSpectralIndex,
    "ckd": CKDSpectralIndex,
}
