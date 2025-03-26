from __future__ import annotations

import attrs
import mitsuba as mi
import pint
import pinttrs

from ._core import BSDF
from ..core import traverse
from ..spectra import Spectrum, spectrum_factory
from ... import validators
from ...attrs import define, documented
from ...kernel import TypeIdLookupStrategy, UpdateParameter
from ...units import unit_registry as ureg


@define(eq=False, slots=False)
class OceanMishchenkoBSDF(BSDF):
    """
    Ocean Mishchenko BSDF [``ocean_mishchenko``].

    This plugin implements the polarized ocean surface model as implemented
    by :cite:t:`Mishchenko1997AerosolRetrievalPolarization`. This model treats
    the ocean as an opaque surface and models the polarized sunglint. It depends
    on wind speed and the index of refraction of the water and external medium
    (assumed to be air).

    See Also
    --------
    :ref:`plugin-bsdf-ocean_legacy`
    """

    wind_speed: pint.Quantity = documented(
        pinttrs.field(
            units=ureg("m/s").units,
            factory=lambda: 0.01 * ureg("m/s"),
            validator=[validators.is_positive, pinttrs.validators.has_compatible_units],
        ),
        doc="Wind speed [m/s] at 10 meters above the surface.",
        type="quantity",
        init_type="quantity or float",
        default="0.01 m/s",
    )

    eta: Spectrum = documented(
        attrs.field(
            default=1.33,
            converter=spectrum_factory.converter("dimensionless"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("dimensionless"),
            ],
        ),
        doc="Real component of the water's index of refraction "
        "processed by :data:`.spectrum_factory`.",
        type=".Spectrum",
        init_type=".Spectrum or dict or float",
        default="1.33",
    )

    k: Spectrum = documented(
        attrs.field(
            default=0.0,
            converter=spectrum_factory.converter("dimensionless"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("dimensionless"),
            ],
        ),
        doc="Imaginary component of the water's index of refraction "
        "processed by :data:`.spectrum_factory`.",
        type=".Spectrum",
        init_type=".Spectrum or dict or float",
        default="0.",
    )

    ext_ior: Spectrum = documented(
        attrs.field(
            default=1.0,
            converter=spectrum_factory.converter("dimensionless"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("dimensionless"),
            ],
        ),
        doc="Real component of the air's index of refraction "
        "processed by :data:`.spectrum_factory`. The imaginary component "
        "is assumed to be zero. ",
        type=".Spectrum",
        init_type=".Spectrum or dict or float",
        default="1.000277f",
    )

    shadowing: bool = documented(
        attrs.field(
            converter=bool,
            validator=attrs.validators.instance_of(bool),
            default=True,
        ),
        doc="Indicates whether evaluation of BRDF computes shadowing and masking.",
        type="bool",
        default="True",
    )

    @property
    def template(self) -> dict:
        # Inherit docstring
        objects = {
            "eta": traverse(self.eta)[0],
            "k": traverse(self.k)[0],
            "ext_ior": traverse(self.ext_ior)[0],
        }

        result = {
            "type": "ocean_mishchenko",
            "wind_speed": self.wind_speed.m_as("m/s"),
            "shadowing": self.shadowing,
        }

        for obj_key, obj_values in objects.items():
            for key, value in obj_values.items():
                result[f"{obj_key}.{key}"] = value

        if self.id is not None:
            result["id"] = self.id

        return result

    @property
    def params(self) -> dict[str, UpdateParameter]:
        # Inherit docstring
        objects = {
            "eta": traverse(self.eta)[1],
            "k": traverse(self.k)[1],
            "ext_ior": traverse(self.ext_ior)[1],
        }

        result = {}
        for obj_key, obj_params in objects.items():
            for key, param in obj_params.items():
                result[f"{obj_key}.{key}"] = attrs.evolve(
                    param,
                    lookup_strategy=TypeIdLookupStrategy(
                        node_type=mi.BSDF,
                        node_id=self.id,
                        parameter_relpath=f"{obj_key}.{key}",
                    )
                    if self.id is not None
                    else None,
                )

        return result
