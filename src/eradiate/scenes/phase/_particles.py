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
    Scattering particle phase function [``particlephase``].

    This class provides an interface to generate kernel dictionaries and define
    scene parameter updates for the phase function associated with scattering
    particles described by the :class:`.ParticleProperties` class.
    """

    particle_properties: ParticleProperties = documented(
        attrs.field(
            validator=attrs.validators.instance_of(ParticleProperties), kw_only=True
        ),
        doc="Scattering property dataset.",
        type="ParticleProperties",
        init_type="ParticleProperties",
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
        """
        Evaluate the scattering angle cosine grid at a given spectral index.

        Parameters
        ----------
        si : .SpectralIndex
            Spectral index.

        Returns
        -------
        ndarray
            Evaluated scattering angle cosine grid.
        """
        return self.eval(si)[0]

    def eval_phase(self, si: SpectralIndex, component: int) -> np.ndarray:
        """
        Evaluate the phase function at a given spectral index, for a specific
        phase matrix component.

        Parameters
        ----------
        si : .SpectralIndex
            Spectral index.

        component : int
            Component index along the ``phamat`` dimension.

        Returns
        -------
        ndarray
            Evaluated phase function as a 1D array.

        Notes
        -----
        Component indices map to phase matrix components as follows:

        * 0 → m11
        * 1 → m12
        * 2 → m33
        * 3 → m34
        * 4 → m22
        * 5 → m44
        """
        return self.eval(si)[1][component, :]

    @singledispatchmethod
    def eval(self, si: SpectralIndex) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate phase function at a given spectral index.

        Parameters
        ----------
        si : .SpectralIndex
            Spectral index.

        Returns
        -------
        mu : ndarray
            Phase angle cosine grid, in ascending order, shape ``(nangles,)``.

        phase : ndarray
            Phase function values for all available phase matrix components,
            shape ``(nphamat, nangles)``.

        Notes
        -----
        * This method dispatches evaluation to specialized methods depending on
          the spectral index type.

        * The default implementation raises an exception.
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
        w : quantity
            Wavelength (must be scalar).

        Returns
        -------
        mu : ndarray
            Phase angle cosine grid, in ascending order, shape ``(nangles,)``.

        phase : ndarray
            Phase function values for all available phase matrix components,
            shape ``(nphamat, nangles)``.
        """
        return self.particle_properties.eval_phase(w=w)

    def eval_ckd(self, w: pint.Quantity, g: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate phase function in ckd modes.

        Parameters
        ----------
        w : quantity
            Spectral bin central wavelength (must be scalar).

        g : float
            Absorption coefficient cumulative probability.

        Returns
        -------
        mu : ndarray
            Phase angle cosine grid, in ascending order, shape ``(nangles,)``.

        phase : ndarray
            Phase function values for all available phase matrix components,
            shape ``(nphamat, nangles)``.
        """
        return self.eval_mono(w=w)

    @property
    def template(self):
        # Inherit docstring
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
        # Inherit docstring
        values_name = "values"

        if self.is_polarized:
            values_name = "m11"

        result = {
            values_name: SceneParameter(
                lambda ctx: self.eval(ctx.si, "11"),
                KernelSceneParameterFlags.SPECTRAL,
            )
        }

        if self.is_polarized:
            if self.particle_properties.has_polarization:
                result["m12"] = SceneParameter(
                    lambda ctx: self.eval(ctx.si, "12"),
                    KernelSceneParameterFlags.SPECTRAL,
                )
                result["m33"] = SceneParameter(
                    lambda ctx: self.eval(ctx.si, "33"),
                    KernelSceneParameterFlags.SPECTRAL,
                )
                result["m34"] = SceneParameter(
                    lambda ctx: self.eval(ctx.si, "34"),
                    KernelSceneParameterFlags.SPECTRAL,
                )

                if self.particle_properties.particle_shape == "spheroidal":
                    result["m22"] = SceneParameter(
                        lambda ctx: self.eval(ctx.si, "22"),
                        KernelSceneParameterFlags.SPECTRAL,
                    )
                    result["m44"] = SceneParameter(
                        lambda ctx: self.eval(ctx.si, "44"),
                        KernelSceneParameterFlags.SPECTRAL,
                    )

                elif self.particle_properties.particle_shape == "spherical":
                    result["m22"] = SceneParameter(
                        lambda ctx: self.eval(ctx.si, "11"),
                        KernelSceneParameterFlags.SPECTRAL,
                    )
                    result["m44"] = SceneParameter(
                        lambda ctx: self.eval(ctx.si, "33"),
                        KernelSceneParameterFlags.SPECTRAL,
                    )

                else:
                    raise NotImplementedError

            else:
                # case: no polarized data but forced polarized. Initialize the
                # diagonal to have the same behaviour as with tabphase in
                # polarized mode.
                result["m22"] = SceneParameter(
                    lambda ctx: self.eval(ctx.si, "11"),
                    KernelSceneParameterFlags.SPECTRAL,
                )

                result["m33"] = SceneParameter(
                    lambda ctx: self.eval(ctx.si, "11"),
                    KernelSceneParameterFlags.SPECTRAL,
                )

                result["m44"] = SceneParameter(
                    lambda ctx: self.eval(ctx.si, "11"),
                    KernelSceneParameterFlags.SPECTRAL,
                )

        return result
