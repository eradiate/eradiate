from __future__ import annotations

import attrs
import mitsuba as mi

from ._core import Illumination
from ..core import traverse
from ..spectra import Spectrum, spectrum_factory
from ...attrs import define, documented
from ...kernel import SceneParameter, SearchSceneParameter
from ...validators import has_quantity


@define(eq=False, slots=False)
class ConstantIllumination(Illumination):
    """
    Constant illumination scene element [``constant``].
    """

    radiance: Spectrum = documented(
        attrs.field(
            default=1.0,
            converter=spectrum_factory.converter("radiance"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                has_quantity("radiance"),
            ],
        ),
        doc="Emitted radiance spectrum. Must be a radiance spectrum "
        "(in W/m²/sr/nm or compatible units).",
        type=":class:`~eradiate.scenes.spectra.Spectrum`",
        init_type=":class:`~eradiate.scenes.spectra.Spectrum` or dict or float",
        default="1.0 ucc[radiance]",
    )

    @property
    def template(self) -> dict:
        result = {"type": "constant", "id": self.id}
        kdict, _ = traverse(self.radiance)
        for k, v in kdict.items():
            result[f"radiance.{k}"] = v
        return result

    @property
    def params(self) -> dict[str, SceneParameter]:
        _, kpmap_radiance = traverse(self.radiance)

        result = {}
        for key, param in kpmap_radiance.items():
            result[f"radiance.{key}"] = attrs.evolve(
                param,
                tracks=SearchSceneParameter(
                    node_type=mi.Emitter,
                    node_id=self.id,
                    parameter_relpath=f"radiance.{param.tracks.strip()}",
                )
                if isinstance(param.tracks, str)
                else param,
            )

        return result
