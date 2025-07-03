from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import attrs

import eradiate

from ..core import NodeSceneElement
from ... import validators
from ..._factory import Factory
from ...attrs import define, documented, get_doc
from ...kernel import DictParameter
from ...spectral.response import DeltaSRF, SpectralResponseFunction
from ...units import unit_registry as ureg

measure_factory = Factory()
measure_factory.register_lazy_batch(
    [
        (
            "_distant.DistantMeasure",
            "distant",
            {},
        ),
        (
            "_distant.MultiPixelDistantMeasure",
            "mpdistant",
            {},
        ),
        (
            "_distant_flux.DistantFluxMeasure",
            "distant_flux",
            {"aliases": ["distantflux"]},
        ),
        (
            "_hemispherical_distant.HemisphericalDistantMeasure",
            "hemispherical_distant",
            {"aliases": ["hdistant"]},
        ),
        (
            "_multi_distant.MultiDistantMeasure",
            "multi_distant",
            {"aliases": ["mdistant"]},
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


def _str_summary_raw(x):
    if not x:
        return "{}"

    keys = list(x.keys())

    if len(keys) == 1:
        return f"dict<1 item>({{{keys[0]}: {{...}} }})"
    else:
        return f"dict<{len(keys)} items>({{{keys[0]}: {{...}} , ... }})"


@define(eq=False, slots=False)
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

    srf: SpectralResponseFunction = documented(
        attrs.field(
            factory=lambda: DeltaSRF(wavelengths=550.0 * ureg.nm),
            converter=SpectralResponseFunction.convert,
            validator=attrs.validators.instance_of(SpectralResponseFunction),
        ),
        doc="Spectral response function (SRF). If a path is passed, it attempts "
        "to load a dataset from that location. If a keyword is passed, *e.g.* "
        "``'sentinel_2a-msi-4'``, the corresponding dataset is looked up "
        "through the file resolver.",
        type=".SpectralResponseFunction",
        init_type="path-like or str or .SpectralResponseFunction or dict",
        default=":class:`DeltaSRF(wavelengths=550.0 * ureg.nm) <.DeltaSRF>`",
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
            "medium.type": DictParameter(
                lambda ctx: (
                    "ref"
                    if f"{self.sensor_id}.atmosphere_medium_id" in ctx.kwargs
                    else DictParameter.UNUSED
                ),
            ),
            "medium.id": DictParameter(
                lambda ctx: (
                    ctx.kwargs[f"{self.sensor_id}.atmosphere_medium_id"]
                    if f"{self.sensor_id}.atmosphere_medium_id" in ctx.kwargs
                    else DictParameter.UNUSED
                ),
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
