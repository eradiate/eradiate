from __future__ import annotations

import enum
import typing as t
import warnings
from abc import ABC, abstractmethod
from collections import abc

import attr
import numpy as np
import pint
import pinttr

import eradiate

from ..core import SceneElement
from ... import ckd, converters, validators
from ..._factory import Factory
from ..._mode import ModeFlags
from ...attrs import AUTO, AutoType, documented, get_doc, parse_docs
from ...ckd import Bin
from ...contexts import CKDSpectralContext, MonoSpectralContext, SpectralContext
from ...exceptions import ModeError, UnsupportedModeError
from ...units import interpret_quantities
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg

measure_factory = Factory()


# ------------------------------------------------------------------------------
#                   Spectral configuration data structures
# ------------------------------------------------------------------------------


class MeasureSpectralConfig(ABC):
    """
    Data structure specifying the spectral configuration of a :class:`.Measure`.

    While this class is abstract, it should however be the main entry point
    to create :class:`.MeasureSpectralConfig` child class objects through the
    :meth:`.MeasureSpectralConfig.new` class method constructor.
    """

    @abstractmethod
    def spectral_ctxs(self) -> t.List[SpectralContext]:
        """
        Return a list of :class:`.SpectralContext` objects based on the
        stored spectral configuration. These data structures can be used to
        drive the evaluation of spectrally dependent components during a
        spectral loop.

        Returns
        -------
        list of :class:`.SpectralContext`
            List of generated spectral contexts. The concrete class
            (:class:`.MonoSpectralContext`, :class:`.CKDSpectralContext`, etc.)
            depends on the active mode.
        """
        pass

    @staticmethod
    def new(**kwargs) -> MeasureSpectralConfig:
        """
        Create a new instance of one of the :class:`.SpectralContext` child
        classes. *The instantiated class is defined based on the currently active
        mode.* Keyword arguments are passed to the instantiated class's
        constructor.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments depending on the currently active mode (see below
            for a list of actual keyword arguments).

        wavelengths : quantity, default: [550] nm
            **Monochromatic modes** [:class:`.MonoMeasureSpectralConfig`].
            List of wavelengths (automatically converted to a Numpy array).
            *Unit-enabled field (default: ucc[wavelength]).*

        bin_set : :class:`.BinSet` or str, default: "10nm"
            **CKD modes** [:class:`.CKDMeasureSpectralConfig`].
            CKD bin set definition. If a string is passed, the data
            repository is queried for the corresponding identifier using
            :meth:`.BinSet.from_db`.

        bins : list of (str or tuple or dict or callable)
            **CKD modes** [:class:`.CKDSpectralContext`].
            List of CKD bins on which to perform the spectral loop. If unset,
            all the bins defined by the selected bin set will be covered.

        See Also
        --------
        :func:`eradiate.mode`, :func:`eradiate.set_mode`
        """
        mode = eradiate.mode()

        if mode is None:
            raise ModeError(
                "instantiating MeasureSpectralConfig requires a mode to be selected"
            )

        if mode.has_flags(ModeFlags.ANY_MONO):
            return MonoMeasureSpectralConfig(**kwargs)

        elif mode.has_flags(ModeFlags.ANY_CKD):
            return CKDMeasureSpectralConfig(**kwargs)

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    @staticmethod
    def from_dict(d: t.Dict) -> MeasureSpectralConfig:
        """
        Create from a dictionary. This class method will additionally pre-process
        the passed dictionary to merge any field with an associated ``"_units"``
        field into a :class:`pint.Quantity` container.

        Parameters
        ----------
        d : dict
            Configuration dictionary used for initialisation.

        Returns
        -------
        :class:`.MeasureSpectralConfig`
            Created object.
        """

        # Pre-process dict: apply units to unit-enabled fields
        d_copy = pinttr.interpret_units(d, ureg=ureg)

        # Perform object creation
        return MeasureSpectralConfig.new(**d_copy)

    @staticmethod
    def convert(value: t.Any) -> t.Any:
        """
        Object converter method.

        If ``value`` is a dictionary, this method uses :meth:`from_dict` to
        create a :class:`.SpectralContext`.

        Otherwise, it returns ``value``.
        """
        if isinstance(value, dict):
            return MeasureSpectralConfig.from_dict(value)

        return value


@parse_docs
@attr.s
class MonoMeasureSpectralConfig(MeasureSpectralConfig):
    """
    A data structure specifying the spectral configuration of a :class:`.Measure`
    in monochromatic modes.
    """

    _wavelengths: pint.Quantity = documented(
        pinttr.ib(
            default=ureg.Quantity([550.0], ureg.nm),
            units=ucc.deferred("wavelength"),
            converter=lambda x: converters.on_quantity(np.atleast_1d)(
                pinttr.converters.ensure_units(x, ucc.get("wavelength"))
            ),
        ),
        doc="List of wavelengths on which to perform the monochromatic spectral "
        "loop.\n\nUnit-enabled field (default: ucc['wavelength']).",
        type="quantity",
        init_type="quantity or array-like",
        default="[550.0] nm",
    )

    @property
    def wavelengths(self):
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, value):
        self._wavelengths = value

    def spectral_ctxs(self) -> t.List[MonoSpectralContext]:
        return [
            MonoSpectralContext(wavelength=wavelength)
            for wavelength in self._wavelengths
        ]


def _ckd_measure_spectral_config_bins_converter(value):
    # Converter for CKDMeasureSpectralConfig.bins
    #

    if isinstance(value, str):
        value = [value]

    bin_set_specs = []
    for bin_set_spec in value:
        # Strings and callables are passed without change
        if isinstance(bin_set_spec, str) or callable(bin_set_spec):
            bin_set_specs.append(bin_set_spec)
            continue

        # In sequences and dicts, we interpret units
        if isinstance(bin_set_spec, abc.Sequence):
            if bin_set_spec[0] == "interval":
                bin_set_specs.append(
                    (
                        "interval",
                        interpret_quantities(
                            bin_set_spec[1],
                            {"wmin": "wavelength", "wmax": "wavelength"},
                            ucc,
                        ),
                    )
                )
            else:
                bin_set_specs.append(bin_set_spec)

            continue

        if isinstance(bin_set_spec, abc.Mapping):
            if bin_set_spec["type"] == "interval":
                bin_set_specs.append(
                    {
                        "type": "interval",
                        "filter_kwargs": interpret_quantities(
                            bin_set_spec["filter_kwargs"],
                            {"wmin": "wavelength", "wmax": "wavelength"},
                            ucc,
                        ),
                    }
                )

            else:
                bin_set_specs.append(bin_set_spec)

            continue

        raise ValueError(f"unhandled CKD bin specification {bin_set_spec}")

    return bin_set_specs


@parse_docs
@attr.s(frozen=True)
class CKDMeasureSpectralConfig(MeasureSpectralConfig):
    """
    A data structure specifying the spectral configuration of a
    :class:`.Measure` in CKD modes.
    """

    # TODO: replace manual bin selection with automation based on sensor spectral
    #  response (with a system to easily design arbitrary SSRs for cases where an
    #  instrument is not simulated)

    bin_set: ckd.BinSet = documented(
        attr.ib(
            default="10nm",
            converter=ckd.BinSet.convert,
            validator=attr.validators.instance_of(ckd.BinSet),
        ),
        doc="CKD bin set definition. If a string is passed, the data "
        "repository is queried for the corresponding identifier using "
        ":meth:`.BinSet.from_db`.",
        type=":class:`~.ckd.BinSet`",
        init_type=":class:`~.ckd.BinSet` or str",
        default='"10nm"',
    )

    _bins: t.Union[t.List[str], AutoType] = documented(
        attr.ib(
            default=AUTO,
            converter=converters.auto_or(_ckd_measure_spectral_config_bins_converter),
        ),
        doc="List of CKD bins on which to perform the spectral loop. If set to "
        "``AUTO``, all the bins relevant to the selected spectral response will be "
        "covered.",
        type="list of str or AUTO",
        init_type="list of (str or tuple or dict or callable) or AUTO",
        default="AUTO",
    )

    @_bins.validator
    def _bins_validator(self, attribute, value):
        if value is AUTO:
            return

        for bin_spec in value:
            if not (
                isinstance(bin_spec, (str, list, tuple, dict)) or callable(bin_spec)
            ):
                raise ValueError(
                    f"while validating {attribute.name}: unsupported CKD bin "
                    f"specification {bin_spec}; expected str, sequence, mapping "
                    f"or callable"
                )

    @property
    def bins(self) -> t.Tuple[Bin]:
        """
        Returns
        -------
        tuple of :class:`.Bin`
            List of selected bins.
        """
        if self._bins is not AUTO:
            bin_selectors = self._bins
        else:
            bin_selectors = [lambda x: True]

        return self.bin_set.select_bins(*bin_selectors)

    def spectral_ctxs(self) -> t.List[CKDSpectralContext]:
        ctxs = []

        for bin in self.bins:
            for bindex in bin.bindexes:
                ctxs.append(CKDSpectralContext(bindex))

        return ctxs


# ------------------------------------------------------------------------------
#                             Measure base class
# ------------------------------------------------------------------------------


class MeasureFlags(enum.Flag):
    """
    Measure flags.
    """

    DISTANT = (
        enum.auto()
    )  #: Measure records radiometric quantities at an infinite distance from the scene.


def _str_summary_raw(x):
    if not x:
        return "{}"

    keys = list(x.keys())
    return f"dict<{len(keys)} items>({{{keys[0]}: {{...}} , ... }})"


@parse_docs
@attr.s
class Measure(SceneElement, ABC):
    """
    Abstract base class for all measure scene elements.

    Notes
    -----
    Raw results stored in the `results` field as nested dictionaries with the
    following structure:

    .. code:: python

       {
           spectral_key_0: dict_0,
           spectral_key_1: dict_1,
           ...
       }

    Keys are spectral loop indexes; values are nested dictionaries produced by
    :func:`.run_mitsuba`.

    See Also
    --------
    :func:`.run_mitsuba`
    """

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    flags: MeasureFlags = documented(
        attr.ib(default=0, converter=MeasureFlags, init=False),
        doc="Measure flags.",
        type=":class:`.MeasureFlags",
    )

    id: t.Optional[str] = documented(
        attr.ib(
            default="measure",
            validator=attr.validators.optional((attr.validators.instance_of(str))),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        init_type=get_doc(SceneElement, "id", "init_type"),
        default='"measure"',
    )

    results: t.Dict = documented(
        attr.ib(factory=dict, repr=_str_summary_raw),
        doc="Storage for raw results yielded by the kernel.",
        type="dict",
        default="{}",
    )

    spectral_cfg: MeasureSpectralConfig = documented(
        attr.ib(
            factory=MeasureSpectralConfig.new,
            converter=MeasureSpectralConfig.convert,
        ),
        doc="Spectral configuration of the measure. Must match the current "
        "operational mode. Can be passed as a dictionary, which will be "
        "interpreted by :meth:`.MeasureSpectralConfig.from_dict`.",
        type=":class:`.MeasureSpectralConfig`",
        init_type=":class:`.MeasureSpectralConfig` or dict",
        default=":meth:`MeasureSpectralConfig.new() <.MeasureSpectralConfig.new>`",
    )

    spp: int = documented(
        attr.ib(default=32, converter=int, validator=validators.is_positive),
        doc="Number of samples per pixel.",
        type="int",
        default="32",
    )

    split_spp: t.Optional[int] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(int),
            validator=attr.validators.optional(validators.is_positive),
        ),
        type="int",
        init_type="int, optional",
        doc="If set, this measure will be split into multiple sensors, each "
        "with a sample count lower or equal to `split_spp`. This parameter "
        "should be used in single-precision modes when the sample count is "
        "higher than 100,000 (very high sample count might result in floating "
        "point number precision issues otherwise).",
    )

    @split_spp.validator
    def _split_spp_validator(self, attribute, value):
        if (
            eradiate.mode().has_flags(ModeFlags.ANY_SINGLE)
            and self.spp > 1e5
            and self.split_spp is None
        ):
            warnings.warn(
                "In single-precision modes, setting a sample count ('spp') to "
                "values greater than 100,000 may result in floating point "
                "precision issues: using the measure's 'split_spp' parameter is "
                "recommended."
            )

    @property
    @abstractmethod
    def film_resolution(self) -> t.Tuple[int, int]:
        """
        tuple: Getter for film resolution as a (int, int) pair.
        """
        pass

    # --------------------------------------------------------------------------
    #                        Kernel dictionary generation
    # --------------------------------------------------------------------------

    def _split_spp_active(self) -> bool:
        """
        Return ``True`` iff sample count split shall be activated.
        """
        return self.split_spp is not None and self.spp > self.split_spp

    def _sensor_id(self, i_spp=None):
        """
        Assemble a sensor ID from indexes on sensor coordinates. This basic
        implementation assumes that the only sensor dimension is ``i_spp``.
        """
        components = [self.id]

        if i_spp is not None:
            components.append(f"spp{i_spp}")

        return "_".join(components)

    def _sensor_spps(self) -> t.List[int]:
        """
        Generate a list of sample counts, possibly accounting for a sample count
        splitting strategy.

        Returns
        -------
        list of int
            List of split sample counts.
        """
        if self._split_spp_active():
            spps = [self.split_spp] * int(self.spp / self.split_spp)

            if self.spp % self.split_spp:
                spps.append(self.spp % self.split_spp)

            return spps

        else:
            return [self.spp]

    def _sensor_ids(self) -> t.List[str]:
        if self.split_spp is not None and self.spp > self.split_spp:
            return [self._sensor_id(i) for i, _ in enumerate(self._sensor_spps())]

        else:
            return [self._sensor_id()]

    # --------------------------------------------------------------------------
    #                        Post-processing information
    # --------------------------------------------------------------------------

    @property
    def var(self) -> str:
        return "img"

    @property
    def sensor_dims(self) -> t.Tuple[str]:
        if self._split_spp_active():
            return ("spp",)
        else:
            return tuple()
