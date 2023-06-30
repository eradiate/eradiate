from __future__ import annotations

import os
import typing as t
import warnings
from abc import ABC, abstractmethod

import attrs
import xarray as xr

import eradiate

from ..core import NodeSceneElement
from ..spectra import (
    InterpolatedSpectrum,
    MultiDeltaSpectrum,
    Spectrum,
    spectrum_factory,
)
from ... import validators
from ..._factory import Factory
from ...attrs import documented, get_doc, parse_docs
from ...kernel import InitParameter
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
    * If ``value`` is a path, the converter tries to open the corresponding
        file on the hard drive; should that fail, it queries the Eradiate data
        store with that path.
    * If ``value`` is a string, it is interpreted as an SRF identifier:
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
        converted = converter(value)
        if not isinstance(converted, (InterpolatedSpectrum, MultiDeltaSpectrum)):
            raise ValueError(
                f"SRF must be an InterpolatedSpectrum or MultiDeltaSpectrum, "
                f"got {converted}"
            )
        return converted


def _str_summary_raw(x):
    if not x:
        return "{}"

    keys = list(x.keys())

    if len(keys) == 1:
        return f"dict<1 item>({{{keys[0]}: {{...}} }})"
    else:
        return f"dict<{len(keys)} items>({{{keys[0]}: {{...}} , ... }})"


@parse_docs
@attrs.define(eq=False, slots=False)
class Measure(NodeSceneElement, ABC):
    """
    Abstract base class for all measure scene elements.

    See Also
    --------
    :func:`.mitsuba_run`

    Notes
    -----
    * This class is meant to be used as a mixin.
    * Raw results stored in the `results` field as nested dictionaries with the
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

    id: str | None = documented(
        attrs.field(
            default="measure",
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc=get_doc(NodeSceneElement, "id", "doc"),
        type=get_doc(NodeSceneElement, "id", "type"),
        init_type=get_doc(NodeSceneElement, "id", "init_type"),
        default='"measure"',
    )

    mi_results: dict = documented(
        attrs.field(factory=dict, repr=_str_summary_raw, init=False),
        doc="Storage for raw results yielded by the kernel.",
        type="dict",
        default="{}",
    )

    srf: Spectrum = documented(
        attrs.field(
            factory=lambda: MultiDeltaSpectrum(wavelengths=550.0 * ureg.nm),
            converter=_srf_converter,
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
        init_type="Path or str or .Spectrum or dict",
        default=":class:`MultiDeltaSpectrum(wavelengths=550.0 * ureg.nm) "
        "<.MultiDeltaSpectrum>`",
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

    rfilter: str = documented(
        attrs.field(
            default="box",
            validator=attrs.validators.in_({"box", "gaussian"}),
        ),
        doc="Reconstruction filter used to scatter samples on sensor pixels. "
        "By default, using a box filter is recommended.",
        type="str",
        init_type='{"box", "gaussian"}',
        default='"box"',
    )

    spp: int = documented(
        attrs.field(default=1000, converter=int, validator=validators.is_positive),
        doc="Number of samples per pixel.",
        type="int",
        default="1000",
    )

    @spp.validator
    def _spp_validator(self, attribute, value):
        if eradiate.mode().is_single_precision and value > 100000:
            warnings.warn(
                f"Measure {self.id} is defined with a sample count greater "
                "than 1e5, but the selected mode is single-precision: results "
                "may be incorrect."
            )

    @property
    @abstractmethod
    def film_resolution(self) -> tuple[int, int]:
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

    # --------------------------------------------------------------------------
    #                        Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def kernel_type(self) -> str:
        raise NotImplementedError

    @property
    def sensor_id(self) -> str:
        return self.id

    @property
    def template(self) -> dict:
        result = {
            "type": self.kernel_type,
            "id": self.sensor_id,
            "film.type": "hdrfilm",
            "film.width": self.film_resolution[0],
            "film.height": self.film_resolution[1],
            "film.pixel_format": "luminance",
            "film.component_format": "float32",
            "film.rfilter.type": "box",
            "sampler.type": self.sampler,
            "sampler.sample_count": self.spp,
            "medium.type": InitParameter(
                lambda ctx: "ref"
                if f"{self.sensor_id}.atmosphere_medium_id" in ctx.kwargs
                else InitParameter.UNUSED,
            ),
            "medium.id": InitParameter(
                lambda ctx: ctx.kwargs[f"{self.sensor_id}.atmosphere_medium_id"]
                if f"{self.sensor_id}.atmosphere_medium_id" in ctx.kwargs
                else InitParameter.UNUSED,
            ),
        }

        return result

    # --------------------------------------------------------------------------
    #                        Post-processing information
    # --------------------------------------------------------------------------

    @property
    def var(self) -> tuple[str, dict]:
        """
        str, dict: Post-processing variable field name and metadata.
        """
        return "img", dict()
