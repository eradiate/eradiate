from __future__ import annotations

from functools import singledispatchmethod
from typing import Any

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

    @singledispatchmethod
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
            Scattering angle cosine grid at the requested spectral index.

        Notes
        -----
        * This method dispatches evaluation to specialized methods depending on
          the spectral index type. The default implementation raises an exception.

        * The underlying implementation caches the output based on the object ID
          of the ``si.w`` argument: this avoids recomputations when calling this
          method repeatedly during a spectral loop iteration.
        """
        raise NotImplementedError

    @eval_mu.register(MonoSpectralIndex)
    def _(self, si) -> np.ndarray:
        result, _ = self._eval_impl(si.w)
        return result

    @eval_mu.register(CKDSpectralIndex)
    def _(self, si) -> np.ndarray:
        result, _ = self._eval_impl(si.w)
        return result

    @singledispatchmethod
    def eval_phase(self, si: SpectralIndex, phamat: int | None = None) -> np.ndarray:
        """
        Evaluate phase function at a given spectral index, for a given phase
        matrix component.

        Parameters
        ----------
        si : .SpectralIndex
            Spectral index.

        phamat : int, optional
            Index of the requested phase matrix coefficient. If unset, all
            coefficients are returned.

        Returns
        -------
        phase : ndarray
            Phase function values for all available phase matrix components,
            shape ``(nphamat,)``.

        Notes
        -----
        * This method dispatches evaluation to specialized methods depending on
          the spectral index type. The default implementation raises an exception.

        * The underlying implementation caches the output based on the object ID
          of the ``si.w`` argument: this avoids recomputations when calling this
          method repeatedly during a spectral loop iteration.

        * Coefficient indices map to phase matrix components as follows:

          * 0 → m11
          * 1 → m12
          * 2 → m33
          * 3 → m34
          * 4 → m22
          * 5 → m44
        """
        raise NotImplementedError

    @eval_phase.register(MonoSpectralIndex)
    def _(self, si, phamat: int | None = None) -> np.ndarray:
        _, result = self._eval_impl(si.w)
        if phamat is not None:
            result = result[phamat, :]
        return result

    @eval_phase.register(CKDSpectralIndex)
    def _(self, si, phamat: int | None = None) -> np.ndarray:
        _, result = self._eval_impl(si.w)
        if phamat is not None:
            result = result[phamat, :]
        return result

    @cache_by_id
    def _eval_impl(self, w: pint.Quantity) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the phase function at a given wavelength.

        Parameters
        ----------
        w : quantity
            Wavelength (must be scalar).

        Returns
        -------
        mu : ndarray
            Scattering angle cosine grid for the requested wavelength, shape
            ``(nangles,)``.

        phase : ndarray
            Evaluated phase function as a 1D array, shape
            ``(nphamat, nangles,)``.
        """
        return self.particle_properties.eval_phase(w=w)

    def _param_to_phamat(self) -> dict[str, int]:
        result = {"m11" if self.is_polarized else "values": 0}

        if self.is_polarized:
            if self.particle_properties.has_polarization:
                result.update({"m12": 1, "m33": 2, "m34": 3})

                if self.particle_properties.particle_shape == "spheroidal":
                    result.update({"m22": 4, "m44": 6})
                elif self.particle_properties.particle_shape == "spherical":
                    result.update({"m22": 0, "m44": 2})

            else:  # not self.particle_properties.has_polarization
                # Case: no polarized data but forced polarized. Initialize the
                # diagonal to have the same behaviour as with tabphase in
                # polarized mode.
                result.update({"m22": 0, "m33": 0, "m44": 0})

        return result

    @property
    def template(self):
        # Inherit docstring

        plugin = "tabphase_polarized" if self.is_polarized else "tabphase_irregular"
        params_to_phamat = self._param_to_phamat()

        result: dict[str, Any] = {"type": plugin}
        result.update(
            {
                k: DictParameter(
                    # Bind v as a default argument to force immediate capture of
                    # current iteration value and avoid a classic Python
                    # closure-in-loop bug
                    lambda ctx, v=v: ",".join(map(str, self.eval_phase(ctx.si, v)))
                )
                for k, v in params_to_phamat.items()
            }
        )

        result["nodes"] = DictParameter(
            lambda ctx: ",".join(map(str, self.eval_mu(ctx.si)))
        )

        return result

    @property
    def params(self) -> dict[str, SceneParameter]:
        # Inherit docstring

        params_to_phamat = self._param_to_phamat()
        result: dict[str, SceneParameter] = {
            k: SceneParameter(
                # Bind v as a default argument to force immediate capture of
                # current iteration value and avoid a classic Python
                # closure-in-loop bug
                lambda ctx, v=v: self.eval_phase(ctx.si, v),
                KernelSceneParameterFlags.SPECTRAL,
            )
            for k, v in params_to_phamat.items()
        }

        result["nodes"] = SceneParameter(
            lambda ctx: self.eval_mu(ctx.si),
            KernelSceneParameterFlags.SPECTRAL,
        )

        return result
