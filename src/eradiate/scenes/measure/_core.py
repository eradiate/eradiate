from __future__ import annotations

import os
import typing as t
import warnings
from abc import ABC, abstractmethod

import attrs
import xarray as xr

import eradiate

from ..core import KernelDict, SceneElement
from ..spectra import InterpolatedSpectrum, Spectrum, UniformSpectrum, spectrum_factory
from ... import validators
from ..._factory import Factory
from ...attrs import documented, get_doc, parse_docs
from ...contexts import KernelDictContext
from ...spectral_index import SpectralIndex
from ...srf_tools import convert as convert_srf
from ...units import PhysicalQuantity
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

def _srf_converter(value: t.Any) -> Spectrum:
    """
    Converter for :class:`.Measure` ``srf`` attribute.

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
    * If ``value`` is a path, the converter tries to open the corresponding "
        file on the hard drive; should that fail, it queries the Eradiate data
        store with that path.
    * If ``value`` is a string, it is interpreted as a SRF identifier:
      * If the identifier does not end with `-raw`, the converter looks for a
        prepared version of the SRF and loads it if it exists, else it loads the
        raw SRF.
      * If the identifier ends with `-raw`, the converter loads the raw SRF.
    """
    if isinstance(value, (str, os.PathLike, xr.Dataset)):
        ds = convert_srf(value)
        w = ureg.Quantity(ds.w.values, ds.w.attrs["units"])
        srf = ds.data_vars["srf"].values
        return InterpolatedSpectrum(quantity="dimensionless", wavelengths=w, values=srf)
    else:
        converter = spectrum_factory.converter(quantity="dimensionless")
        return converter(value)

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

    srf: Spectrum = documented(
        attrs.field(
            factory=lambda: UniformSpectrum(value=1.0),
            converter=_srf_converter,
            validator=validators.has_quantity(PhysicalQuantity.DIMENSIONLESS),  # TODO: supports UniformSpectrum and InterpolatedSpectrum and DisceteSpectrum
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

    results: t.Dict = documented(
        attrs.field(factory=dict, repr=_str_summary_raw, init=False),
        doc="Storage for raw results yielded by the kernel.",
        type="dict",
        default="{}",
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
