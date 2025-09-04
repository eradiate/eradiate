"""
Spectral index data structure.

A spectral index data structure holds the information required to evaluate
spectral quantities. The simplest example is provided by
:class:`MonoSpectralIndex` which holds a single wavelength value.
Each SpectralIndex subclass is specific to an Eradiate spectral mode.
For example, :class:`MonoSpectralIndex` is used in monochromatic mode.

.. admonition:: Note to maintainers
   :class: note

   All methods that evaluate spectral quantities, conventionally named ``eval_*``,
   should accept a ``si: SpectralIndex`` parameter.

   If a new spectral mode is added, you need to:

   * add a new :class:`.SpectralIndex` subclass
   * add it to the :data:`.SPECTRAL_MODE_DISPATCH` dictionary, which maps
     spectral mode enum values to the corresponding spectral index type

   For experiments to support the newly added spectral mode, you will also need to
   update the ``eradiate/src/experiment`` package, in particular the
   ``_normalize_spectral()`` post-init method of the
   :class:`.EarthObservationExperiment`.
"""

from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

import attrs
import pint
import pinttr

from .. import validators
from .._mode import ModeFlag, SubtypeDispatcher
from ..attrs import define, documented
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg


@attrs.define(eq=False, frozen=True, slots=True)
class SpectralIndex(ABC):
    """
    Abstract spectral index base class.

    See Also
    --------
    :class:`.MonoSpectralIndex`, :class:`.CKDSpectralIndex`
    """

    subtypes = SubtypeDispatcher("SpectralIndex")

    @property
    @abstractmethod
    def formatted_repr(self) -> str:
        """
        Formatted representation of the spectral index.
        """
        pass

    @property
    @abstractmethod
    def as_hashable(self) -> t.Hashable:
        """
        Hashable representation of the spectral index.

        Notes
        -----
        The hashable value is used by :class:`eradiate.experiments.Experiment`
        to identify the simulation results corresponding to a given spectral
        index.
        """
        pass

    @staticmethod
    def new(mode: ModeFlag | str | None = None, **kwargs) -> SpectralIndex:
        """
        Spectral index factory method.

        Parameters
        ----------
        mode : `.ModeFlag` or str, optional
            Spectral mode identifier. If ``None``, the current mode is used.

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
        :class:`.MonoSpectralIndex`, :class:`.CKDSpectralIndex`
        """
        if isinstance(mode, str):
            mode = mode.upper()
            if not mode.startswith("SPECTRAL_MODE_"):
                mode = f"SPECTRAL_MODE_{mode}"
            mode = ModeFlag[mode]
        si_cls = SpectralIndex.subtypes.resolve(mode)
        return si_cls(**kwargs)

    @staticmethod
    def from_dict(d: dict[str, t.Any]) -> SpectralIndex:
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


@SpectralIndex.subtypes.register(ModeFlag.SPECTRAL_MODE_MONO)
@define(eq=False, frozen=True, slots=True)
class MonoSpectralIndex(SpectralIndex):
    """
    Monochromatic spectral index.

    See Also
    --------
    :class:`CKDSpectralIndex`
    """

    w: pint.Quantity = documented(
        pinttr.field(
            default=550.0 * ureg.nm,
            units=ucc.deferred("wavelength"),
            on_setattr=None,
        ),
        doc="Wavelength.\n\nUnit-enabled field (default: ucc[wavelength]).",
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


@SpectralIndex.subtypes.register(ModeFlag.SPECTRAL_MODE_CKD)
@define(eq=False, frozen=True, slots=True)
class CKDSpectralIndex(SpectralIndex):
    """
    CKD spectral index.

    See Also
    --------
    :class:`MonoSpectralIndex`
    """

    w: pint.Quantity = documented(
        pinttr.field(
            default=550.0 * ureg.nm,
            units=ucc.deferred("wavelength"),
            on_setattr=None,
        ),
        doc="Bin center wavelength.\n\nUnit-enabled field (default: ucc[wavelength]).",
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
        return f"{self.w:g~P}:{self.g:.3f}"

    @property
    def as_hashable(self) -> t.Tuple[float, float]:
        return (float(self.w.m_as(ureg.nm)), self.g)
