from __future__ import annotations

import enum
import os
import typing as t
import warnings
from abc import ABC, abstractmethod
from collections import abc

import attrs
import numpy as np
import pint
import pinttr
import xarray as xr

import eradiate

from ..core import KernelDict, SceneElement
from ..spectra import InterpolatedSpectrum, Spectrum, UniformSpectrum, spectrum_factory
from ... import ckd, converters, validators
from ..._factory import Factory
from ...attrs import AUTO, AutoType, documented, get_doc, parse_docs
from ...ckd import Bin
from ...contexts import (
    CKDSpectralContext,
    KernelDictContext,
    MonoSpectralContext,
    SpectralContext,
)
from ...exceptions import DataError, ModeError, UnsupportedModeError
from ...units import PhysicalQuantity, interpret_quantities
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg

measure_factory = Factory()
measure_factory.register_lazy_batch(
    [
        (
            "_distant_flux.DistantFluxMeasure",
            "distant_flux",
            {},
        ),
        (
            "_hemispherical_distant.HemisphericalDistantMeasure",
            "hemispherical_distant",
            {"aliases": ["hdistant"]},
        ),
        (
            "_multi_distant.MultiDistantMeasure",
            "multi_distant",
            {"aliases": ["mdistant", "distant"]},
        ),
        (
            "_multi_radiancemeter.MultiRadiancemeterMeasure",
            "multi_radiancemeter",
            {"aliases": ["mradiancemeter"]},
        ),
        (
            "_perspective.PerspectiveCameraMeasure",
            "perspective",
            {},
        ),
        (
            "_radiancemeter.RadiancemeterMeasure",
            "radiancemeter",
            {},
        ),
    ],
    cls_prefix="eradiate.scenes.measure",
)


# ------------------------------------------------------------------------------
#                   Spectral configuration data structures
# ------------------------------------------------------------------------------


def _load_srf_ds(value: t.Union[str, os.PathLike]) -> xr.Dataset:
    # path (local or in data store)
    if str(value).endswith(".nc"):
        return converters.load_dataset(value)

    # identifier for a file in the data store
    if isinstance(value, str):
        try:
            return converters.load_dataset(f"spectra/srf/{value}.nc")
        except DataError as e:
            if value.endswith("-raw"):
                raise e
            else:
                warnings.warn(
                    f"Could not serve SRF '{value}' from the data store."
                    f"Trying to serve '{value}-raw' instead."
                )
                return converters.load_dataset(f"spectra/srf/{value}-raw.nc")

    # this is an abnormal state
    raise ValueError(f"unhandled value '{value}'")


def _measure_spectral_config_srf_converter(value: t.Any) -> Spectrum:
    """
    Converter for :class:`MeasureSpectralConfig` ``srf`` attribute.

    Parameters
    ----------
    value : Any
        Value to convert.

    Returns
    -------
    Spectrum
        Converted value.

    Notes
    -----
    The behaviour of this converter depends on the value type:
    * If ``value`` is not a string or path, it is passed to the
      :class:`.spectrum_factory`'s converter.
    * If ``value`` is a path, the corresponding file is loaded
    * If ``value`` is a string, it is interpreted as a SRF identifier:
      * If the identifier does not end with `-raw`, the converter looks for a
        prepared version of the SRF and loads it if it exists, else it loads the
        raw SRF.
      * If the identifier ends with `-raw`, the converter loads the raw SRF.
    """
    if isinstance(value, (str, os.PathLike)):
        ds = _load_srf_ds(value)
        w = ureg.Quantity(ds.w.values, ds.w.attrs["units"])
        srf = ds.data_vars["srf"].values
        return InterpolatedSpectrum(quantity="dimensionless", wavelengths=w, values=srf)
    else:
        converter = spectrum_factory.converter(quantity="dimensionless")
        return converter(value)


@parse_docs
@attrs.define
class MeasureSpectralConfig(ABC):
    """
    Data structure specifying the spectral configuration of a :class:`.Measure`.

    While this class is abstract, it should however be the main entry point
    to create :class:`.MeasureSpectralConfig` child class objects through the
    :meth:`.MeasureSpectralConfig.new` class method constructor.
    """

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    srf: Spectrum = documented(
        attrs.field(
            factory=lambda: UniformSpectrum(value=1.0),
            converter=_measure_spectral_config_srf_converter,
            validator=validators.has_quantity(PhysicalQuantity.DIMENSIONLESS),
        ),
        doc="Spectral response function (SRF). If a path is passed, it attempts "
        "to load a dataset from that location. If a keyword is passed, e.g., "
        "``'sentinel_2a-msi-4'`` it tries to serve the corresponding dataset "
        "from the Eradiate data store. By default, the *prepared* version of "
        "the SRF is served unless it does not exist in which case the *raw* "
        "version is served. To request that the raw version is served, append "
        "``'-raw'`` to the keyword, e.g., ``'sentinel_2a-msi-4-raw'``. "
        "Note that the prepared SRF provide a better speed versus accuracy "
        "trade-off, but for the best accuracy, the raw SRF should be used. "
        "Other types will be converted by :data:`.spectrum_factory`.",
        type=".Spectrum",
        init_type="Path or str or .Spectrum or dict or float",
        default=":class:`UniformSpectrum(value=1.0) <.UniformSpectrum>`",
    )

    # --------------------------------------------------------------------------
    #                         Spectral context generation
    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------
    #                         Class method constructors
    # --------------------------------------------------------------------------

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

        if mode.is_mono:
            return MonoMeasureSpectralConfig(**kwargs)

        elif mode.is_ckd:
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
        spectral_cfg : .MeasureSpectralConfig
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
@attrs.define
class MonoMeasureSpectralConfig(MeasureSpectralConfig):
    """
    A data structure specifying the spectral configuration of a :class:`.Measure`
    in monochromatic modes.
    """

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    _wavelengths: pint.Quantity = documented(
        pinttr.field(
            default=ureg.Quantity([550.0], ureg.nm),
            units=ucc.deferred("wavelength"),
            converter=lambda x: converters.on_quantity(np.atleast_1d)(
                pinttr.util.ensure_units(x, ucc.get("wavelength"))
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

    def __attrs_post_init__(self):
        # Special post-init check to ensure that wavelengths and spectral
        # response are compatible
        srf = self.srf.eval_mono(self.wavelengths)

        if np.allclose(srf, 0.0):
            raise ValueError(
                "specified spectral response function evaluates to 0 at every "
                "selected wavelength"
            )

    # --------------------------------------------------------------------------
    #                         Spectral context generation
    # --------------------------------------------------------------------------

    def spectral_ctxs(self) -> t.List[MonoSpectralContext]:
        return [
            MonoSpectralContext(wavelength=wavelength)
            for wavelength in self._wavelengths
            if not np.isclose(self.srf.eval_mono(wavelength), 0.0)
        ]


def _ckd_measure_spectral_config_bins_converter(value):
    # Converter for CKDMeasureSpectralConfig.bins
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


def _active(bin: Bin, srf: Spectrum) -> bool:
    """
    Test whether a spectral bin is selected by a spectral response function.

    Parameters
    ----------
    bin : .Bin
        CKD bin to test.

    srf : .Spectrum
        Spectral response function.

    Returns
    -------
    bool
        ``True`` if the bin partially or totally overlaps the support of the SRF
        (*i.e.* where it is not zero), ``False`` otherwise.

    Notes
    -----
    Here, "`support <https://en.wikipedia.org/wiki/Support_(mathematics)>`_"
    means the domain where the function takes nonzero values.
    It can be compact (*i.e.* in a single piece) or not (*i.e.* have "holes" or
    be infinite).
    """
    return bool(srf.integral(bin.wmin, bin.wmax) > 0.0)


@parse_docs
@attrs.frozen
class CKDMeasureSpectralConfig(MeasureSpectralConfig):
    """
    A data structure specifying the spectral configuration of a
    :class:`.Measure` in CKD modes.
    """

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    bin_set: ckd.BinSet = documented(
        attrs.field(
            default="10nm",
            converter=ckd.BinSet.convert,
            validator=attrs.validators.instance_of(ckd.BinSet),
        ),
        doc="CKD bin set definition. If a string is passed, the data "
        "repository is queried for the corresponding identifier using "
        ":meth:`.BinSet.from_db`.",
        type=":class:`~.ckd.BinSet`",
        init_type=":class:`~.ckd.BinSet` or str",
        default='"10nm"',
    )

    _bins: t.Union[t.List[str], AutoType] = documented(
        attrs.field(
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
        if self._bins is AUTO:
            # Use the SRF
            bin_selectors = [lambda x: _active(x, self.srf)]
        else:
            # Select bins manually using the selector syntax
            bin_selectors = self._bins

        return self.bin_set.select_bins(*bin_selectors)

    # --------------------------------------------------------------------------
    #                         Spectral context generation
    # --------------------------------------------------------------------------

    def spectral_ctxs(self) -> t.List[CKDSpectralContext]:
        ctxs = []
        bin_set = self.bin_set

        for bin in self.bins:
            for bindex in bin.bindexes:
                ctxs.append(CKDSpectralContext(bindex, bin_set))

        return ctxs


# ------------------------------------------------------------------------------
#                             Measure base class
# ------------------------------------------------------------------------------


def _str_summary_raw(x):
    if not x:
        return "{}"

    keys = list(x.keys())

    if len(keys) == 1:
        return f"dict<1 item>({{{keys[0]}: {{...}} }})"
    else:
        return f"dict<{len(keys)} items>({{{keys[0]}: {{...}} , ... }})"


@parse_docs
@attrs.define
class Measure(SceneElement, ABC):
    """
    Abstract base class for all measure scene elements.

    See Also
    --------
    :func:`.mitsuba_run`

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
    :func:`.mitsuba_run`.
    """

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    id: t.Optional[str] = documented(
        attrs.field(
            default="measure",
            validator=attrs.validators.optional((attrs.validators.instance_of(str))),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        init_type=get_doc(SceneElement, "id", "init_type"),
        default='"measure"',
    )

    results: t.Dict = documented(
        attrs.field(factory=dict, repr=_str_summary_raw, init=False),
        doc="Storage for raw results yielded by the kernel.",
        type="dict",
        default="{}",
    )

    spectral_cfg: MeasureSpectralConfig = documented(
        attrs.field(
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

    sampler: str = documented(
        attrs.field(
            default="independent",
            validator=attrs.validators.in_(
                {"independent", "stratified", "multijitter", "orthogonal", "ldsampler"}
            ),
        ),
        doc="Mitsuba sampler used to generate pseudo-random number sequences.",
        type="str",
        init_type='{"independent", "stratified", "multijitter", "orthogonal", '
        '"ldsampler"}',
        default='"independent"',
    )

    spp: int = documented(
        attrs.field(default=1000, converter=int, validator=validators.is_positive),
        doc="Number of samples per pixel.",
        type="int",
        default="1000",
    )

    split_spp: t.Optional[int] = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(int),
            validator=attrs.validators.optional(validators.is_positive),
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
            eradiate.mode().is_single_precision
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
    #                             Flag-style queries
    # --------------------------------------------------------------------------

    def is_distant(self) -> bool:
        """
        Return ``True`` iff measure records radiometric quantities at infinite
        distance.
        """
        # Default implementation returns False
        return False

    def is_split(self) -> bool:
        """
        Return ``True`` iff sample count split shall be activated.
        """
        return self.split_spp is not None and self.spp > self.split_spp

    # --------------------------------------------------------------------------
    #                        Kernel dictionary generation
    # --------------------------------------------------------------------------

    def _sensor_id(self, i_spp=None) -> str:
        """
        Assemble a sensor ID from indexes on sensor coordinates. This basic
        implementation assumes that the only sensor dimension is ``i_spp``.

        Returns
        -------
        str
            Generated sensor ID.
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
        if self.is_split():
            spps = [self.split_spp] * int(self.spp / self.split_spp)

            if self.spp % self.split_spp:
                spps.append(self.spp % self.split_spp)

            return spps

        else:
            return [self.spp]

    def _sensor_ids(self) -> t.List[str]:
        """
        Return list of sensor IDs for the current measure.

        Returns
        -------
        list of str
            List of sensor IDs.
        """
        if self.split_spp is not None and self.spp > self.split_spp:
            return [self._sensor_id(i) for i, _ in enumerate(self._sensor_spps())]

        else:
            return [self._sensor_id()]

    @KernelDictContext.DYNAMIC_FIELDS.register("atmosphere_medium_id")
    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        # Inherit docstring
        sensor_ids = self._sensor_ids()
        sensor_spps = self._sensor_spps()
        result = KernelDict()

        for spp, sensor_id in zip(sensor_spps, sensor_ids):
            result_dict = self._kernel_dict_impl(sensor_id, spp)

            try:
                result_dict["medium"] = {
                    "type": "ref",
                    "id": ctx.atmosphere_medium_id,
                }
            except AttributeError:
                pass

            result.data[sensor_id] = result_dict

        return result

    @abstractmethod
    def _kernel_dict_impl(self, sensor_id, spp):
        # Implementation of the kernel dict generation routine
        pass

    # --------------------------------------------------------------------------
    #                        Post-processing information
    # --------------------------------------------------------------------------

    @property
    def var(self) -> t.Tuple[str, t.Dict]:
        """
        str, dict: Post-processing variable field name and metadata.
        """
        return "img", dict()

    @property
    def sensor_dims(self) -> t.Tuple[str]:
        """
        tuple of str: List of sensor dimension labels.
        """
        if self.is_split():
            return ("spp",)
        else:
            return tuple()
