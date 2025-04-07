from __future__ import annotations

import attrs
import mitsuba as mi
import pint
import pinttr

from ._core import BSDF
from ..core import traverse
from ..spectra import Spectrum, spectrum_factory
from ... import validators
from ...attrs import define, documented
from ...kernel import SceneParameter, SearchSceneParameter
from ...units import unit_context_config as ucc


@define(eq=False, slots=False)
class RTLSBSDF(BSDF):
    """
    RTLS BSDF [``rtls``].

    This class implements the RossThick-LiSparse (RTLS) BRDF as described
    by the MODIS BRDF/Albedo Product ATBD :cite:`BU-MODISBRDFAlbedoProductATBD1999`.

    See Also
    --------
    :ref:`plugin-bsdf-rtls`
    """

    f_iso: Spectrum = documented(
        attrs.field(
            default=0.209741,
            converter=spectrum_factory.converter("dimensionless"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("dimensionless"),
            ],
        ),
        doc="Isotropic scattering kernel parameter. "
        r"Defaults to :math:`f_{iso} = 0.209741`.",
        type=".Spectrum",
        init_type=".Spectrum or dict or float, optional",
        default="0.209741",
    )

    f_vol: Spectrum = documented(
        attrs.field(
            default=0.004140,
            converter=spectrum_factory.converter("dimensionless"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("dimensionless"),
            ],
        ),
        doc="Volumetric scattering from horizontally homogeneous leaf canopies "
        r"kernel parameter. Defaults to :math:`f_{vol} = 0.004140`.",
        type=".Spectrum",
        init_type=".Spectrum or dict or float, optional",
        default="0.004140",
    )

    f_geo: Spectrum | None = documented(
        attrs.field(
            default=0.081384,
            converter=attrs.converters.optional(
                spectrum_factory.converter("dimensionless")
            ),
            validator=attrs.validators.optional(
                [
                    attrs.validators.instance_of(Spectrum),
                    validators.has_quantity("dimensionless"),
                ]
            ),
        ),
        doc="Geometric-optical surface scattering kernel parameter. "
        r"Defaults to :math:`f_{geo} = 0.081384`.",
        type="float",
        init_type="float, optional",
        default="0.081384",
    )

    h: pint.Quantity = documented(
        pinttr.field(
            default=2.0,
            units=ucc.deferred("dimensionless"),
        ),
        doc="Height-to-center-of-crown. Must be dimensionless.",
        type="quantity",
        init_type="quantity or float",
        default="2.0",
    )

    r: pint.Quantity = documented(
        pinttr.field(
            default=1.0,
            units=ucc.deferred("dimensionless"),
        ),
        doc="Crown horizontal radius. Must not be zero.",
        type="quantity",
        init_type="quantity or float",
        default="1.0",
    )

    b: pint.Quantity = documented(
        pinttr.field(
            default=1.0,
            units=ucc.deferred("dimensionless"),
        ),
        doc="Crown vertical radius. Must not be zero.",
        type="quantity",
        init_type="quantity or float",
        default="1.0",
    )

    @r.validator
    def _r_validator(self, attribute, value):
        assert value != 0.0

    @b.validator
    def _b_validator(self, attribute, value):
        assert value != 0.0

    @property
    def template(self) -> dict:
        # Inherit docstring
        objects = {
            "f_iso": traverse(self.f_iso)[0],
            "f_vol": traverse(self.f_vol)[0],
            "f_geo": traverse(self.f_geo)[0],
        }

        result = {
            "type": "rtls",
            "h": self.h.magnitude,
            "r": self.r.magnitude,
            "b": self.b.magnitude,
        }

        for obj_key, obj_values in objects.items():
            for key, value in obj_values.items():
                result[f"{obj_key}.{key}"] = value

        if self.id is not None:
            result["id"] = self.id

        return result

    @property
    def params(self) -> dict[str, SceneParameter]:
        # Inherit docstring
        objects = {
            "f_iso": traverse(self.f_iso)[1],
            "f_vol": traverse(self.f_vol)[1],
            "f_geo": traverse(self.f_geo)[1],
        }

        result = {}
        for obj_key, obj_params in objects.items():
            for key, param in obj_params.items():
                result[f"{obj_key}.{key}"] = attrs.evolve(
                    param,
                    search=SearchSceneParameter(
                        node_type=mi.BSDF,
                        node_id=self.id,
                        parameter_relpath=f"{obj_key}.{key}",
                    )
                    if self.id is not None
                    else None,
                )

        return result
