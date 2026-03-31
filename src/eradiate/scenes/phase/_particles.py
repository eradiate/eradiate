from __future__ import annotations

from functools import singledispatchmethod

import attrs
import numpy as np
import pint

import eradiate

from ._core import PhaseFunction
from ...attrs import define, documented
from ...kernel import DictParameter, KernelSceneParameterFlags, SceneParameter
from ...radprops._particles import ParticleProperties
from ...spectral import CKDSpectralIndex, MonoSpectralIndex, SpectralIndex
from ...util.misc import cache_by_id


@define(eq=False, slots=False)
class ParticlePhaseFunction(PhaseFunction):
    """
    Scattering particle phase function [``particlephase``]
    """

    particle_properties: ParticleProperties = documented(
        attrs.field(
            validator=attrs.validators.instance_of(ParticleProperties), kw_only=True
        )
    )

    force_polarized: bool = documented(
        attrs.field(default=False, converter=bool, kw_only=True),
        doc="Flag that forces the use of a polarized phase function.",
        type="bool",
        init_type="bool",
        default="False",
    )

    @property
    def is_polarized(self) -> bool:
        return eradiate.get_mode().is_polarized and (
            self.particle_properties.has_polarization or self.force_polarized
        )

    def eval_mu(self, si: SpectralIndex) -> np.ndarray:
        return self.eval(si)[0]

    def eval_phase(self, si: SpectralIndex) -> np.ndarray:
        return self.eval(si)[1]

    @singledispatchmethod
    def eval(self, si: SpectralIndex) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate phase function at a given spectral index.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        Returns
        -------
        ndarray
            Evaluated phase function as a 1D array.

        Notes
        -----
        This method dispatches evaluation to specialized methods depending on
        the spectral index type.
        """
        raise NotImplementedError

    @cache_by_id
    @eval.register(MonoSpectralIndex)
    def _(self, si) -> tuple[np.ndarray, np.ndarray]:
        return self.eval_mono(w=si.w)

    @cache_by_id
    @eval.register(CKDSpectralIndex)
    def _(self, si) -> tuple[np.ndarray, np.ndarray]:
        return self.eval_ckd(w=si.w, g=si.g)

    def eval_mono(self, w: pint.Quantity) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate phase function in monochromatic modes.

        Parameters
        ----------
        w : :class:`pint.Quantity`
            Wavelength.

        Returns
        -------
        ndarray
            Evaluated phase function as a 1D or 2D array depending on the shape
            of `w` (angle dimension comes last).
        """
        return self.particle_properties.eval_phase(w=w)

    def eval_ckd(self, w: pint.Quantity, g: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate phase function in ckd modes.

        Parameters
        ----------
        w : :class:`pint.Quantity`
            Spectral bin central wavelength.

        g : float
            Absorption coefficient cumulative probability.

        Returns
        -------
        ndarray
            Evaluated phase function as a 1D or 2D array depending on the shape
            of `w` (angle dimension comes last).
        """
        return self.eval_mono(w=w)

    @property
    def template(self):
        phase_function = "tabphase_irregular"
        values_name = "values"

        if self.is_polarized:
            phase_function = "tabphase_polarized"
            values_name = "m11"

        result = {
            "type": phase_function,
            values_name: DictParameter(
                lambda ctx: ",".join(map(str, self.eval(ctx.si, "11"))),
            ),
        }

        if self.is_polarized:
            if self.particle_properties.has_polarization:
                result["m12"] = DictParameter(
                    lambda ctx: ",".join(map(str, self.eval(ctx.si, "12"))),
                )
                result["m33"] = DictParameter(
                    lambda ctx: ",".join(map(str, self.eval(ctx.si, "33"))),
                )
                result["m34"] = DictParameter(
                    lambda ctx: ",".join(map(str, self.eval(ctx.si, "34"))),
                )

                if self.particle_properties.particle_shape == "spheroidal":
                    result["m22"] = DictParameter(
                        lambda ctx: ",".join(map(str, self.eval(ctx.si, "22"))),
                    )

                    result["m44"] = DictParameter(
                        lambda ctx: ",".join(map(str, self.eval(ctx.si, "44"))),
                    )

                elif self.particle_properties.particle_shape == "spherical":
                    result["m22"] = DictParameter(
                        lambda ctx: ",".join(map(str, self.eval(ctx.si, "11"))),
                    )

                    result["m44"] = DictParameter(
                        lambda ctx: ",".join(map(str, self.eval(ctx.si, "33"))),
                    )

                else:
                    raise NotImplementedError

            else:
                # case: no polarized data but forced polarized. Initialize the
                # diagonal to have the same behaviour as with tabphase in
                # polarized mode.
                result["m22"] = DictParameter(
                    lambda ctx: ",".join(map(str, self.eval(ctx.si, "11"))),
                )

                result["m33"] = DictParameter(
                    lambda ctx: ",".join(map(str, self.eval(ctx.si, "11"))),
                )

                result["m44"] = DictParameter(
                    lambda ctx: ",".join(map(str, self.eval(ctx.si, "11"))),
                )

        result["nodes"] = DictParameter(
            lambda ctx: ",".join(map(str, self.eval(ctx.si, "11")))
        )

        return result

    @property
    def params(self) -> dict[str, SceneParameter]:
        values_name = "values"

        if self.is_polarized:
            values_name = "m11"

        result = {
            values_name: SceneParameter(
                lambda ctx: self.eval(ctx.si, 0, 0),
                KernelSceneParameterFlags.SPECTRAL,
            )
        }

        if self.is_polarized:
            if self.particle_properties.has_polarized_data:
                result["m12"] = SceneParameter(
                    lambda ctx: self.eval(ctx.si, 0, 1),
                    KernelSceneParameterFlags.SPECTRAL,
                )
                result["m33"] = SceneParameter(
                    lambda ctx: self.eval(ctx.si, 2, 2),
                    KernelSceneParameterFlags.SPECTRAL,
                )
                result["m34"] = SceneParameter(
                    lambda ctx: self.eval(ctx.si, 2, 3),
                    KernelSceneParameterFlags.SPECTRAL,
                )

                if self.particle_properties.particle_shape == "spheroidal":
                    result["m22"] = SceneParameter(
                        lambda ctx: self.eval(ctx.si, 1, 1),
                        KernelSceneParameterFlags.SPECTRAL,
                    )
                    result["m44"] = SceneParameter(
                        lambda ctx: self.eval(ctx.si, 3, 3),
                        KernelSceneParameterFlags.SPECTRAL,
                    )

                elif self.particle_properties.particle_shape == "spherical":
                    result["m22"] = SceneParameter(
                        lambda ctx: self.eval(ctx.si, 0, 0),
                        KernelSceneParameterFlags.SPECTRAL,
                    )
                    result["m44"] = SceneParameter(
                        lambda ctx: self.eval(ctx.si, 2, 2),
                        KernelSceneParameterFlags.SPECTRAL,
                    )

                else:
                    raise NotImplementedError

            else:
                # case: no polarized data but forced polarized. Initialize the
                # diagonal to have the same behaviour as with tabphase in
                # polarized mode.
                result["m22"] = SceneParameter(
                    lambda ctx: self.eval(ctx.si, 0, 0),
                    KernelSceneParameterFlags.SPECTRAL,
                )

                result["m33"] = SceneParameter(
                    lambda ctx: self.eval(ctx.si, 0, 0),
                    KernelSceneParameterFlags.SPECTRAL,
                )

                result["m44"] = SceneParameter(
                    lambda ctx: self.eval(ctx.si, 0, 0),
                    KernelSceneParameterFlags.SPECTRAL,
                )

        return result
