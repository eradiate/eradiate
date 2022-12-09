from __future__ import annotations

import typing as t
import warnings

import attrs
import numpy as np

import eradiate

from ._core import EarthObservationExperiment, Experiment
from ..attrs import AUTO, documented, get_doc, parse_docs
from ..contexts import KernelDictContext
from ..radprops._afgl1986 import G16
from ..scenes.atmosphere import (
    Atmosphere,
    AtmosphereGeometry,
    HeterogeneousAtmosphere,
    HomogeneousAtmosphere,
    MolecularAtmosphere,
    PlaneParallelGeometry,
    SphericalShellGeometry,
    atmosphere_factory,
)
from ..scenes.bsdfs import BSDF, LambertianBSDF, bsdf_factory
from ..scenes.core import KernelDict
from ..scenes.integrators import Integrator, VolPathIntegrator, integrator_factory
from ..scenes.measure import Measure, MultiRadiancemeterMeasure, TargetPoint
from ..scenes.measure._distant import DistantMeasure
from ..scenes.shapes import RectangleShape, SphereShape
from ..scenes.surface import BasicSurface, DEMSurface, surface_factory
from ..spectral_index import CKDSpectralIndex, MonoSpectralIndex, SpectralIndex
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg
from ..util.deprecation import substitute


def measure_inside_atmosphere(atmosphere, measure, ctx):
    """
    Evaluate whether a sensor is placed within an atmosphere.

    Raises a ValueError if called with a :class:`.MultiRadiancemeterMeasure`
    with origins both inside and outside the atmosphere.
    """
    if atmosphere is None:
        return False

    shape = atmosphere.eval_shape(ctx)

    if isinstance(measure, MultiRadiancemeterMeasure):
        inside = shape.contains(measure.origins)

        if all(inside):
            return True
        elif not any(inside):
            return False
        else:
            raise ValueError(
                "Inconsistent placement of MultiRadiancemeterMeasure origins. "
                "Origins must lie either all inside or all outside of the "
                "atmosphere."
            )

    elif isinstance(measure, DistantMeasure):
        # Note: This will break if the user makes something weird such as using
        # a large offset value which would put some origins outside and others
        # inside the atmosphere shape
        return not measure.is_distant()

    else:
        # Note: This will likely break if a new measure type is added
        return shape.contains(measure.origin)


def _surface_converter(value):
    if isinstance(value, dict):
        try:
            # First, attempt conversion to BSDF
            value = bsdf_factory.convert(value)
        except TypeError:
            # If this doesn't work, attempt conversion to Surface
            return surface_factory.convert(value)

    # If we make it to this point, it means that dict conversion has been
    # performed with success
    if isinstance(value, BSDF):
        return BasicSurface(
            shape=RectangleShape(),
            bsdf=value,
        )

    return value


@parse_docs
@attrs.define
class AtmosphereExperiment(EarthObservationExperiment):
    """
    Simulate radiation in a one-dimensional scene. This experiment approximates
    a one-dimensional setup using a 3D geometry set up to reproduce the
    effect of invariances typical of 1D geometries. It supports the so-called
    plane parallel and spherical shell geometries.

    Notes
    -----
    * A post-initialisation step will constrain the measure setup if a
      distant measure is used and no target is defined:

      * if an atmosphere is defined, the target will be set to [0, 0, TOA];
      * if no atmosphere is defined, the target will be set to [0, 0, 0].

    * This experiment supports arbitrary measure positioning, except for
      :class:`.MultiRadiancemeterMeasure`, for which subsensor origins are
      required to be either all inside or all outside of the atmosphere. If an
      unsuitable configuration is detected, a :class:`ValueError` will be raised
      during initialisation.
    """

    geometry: t.Union[PlaneParallelGeometry, SphericalShellGeometry] = documented(
        attrs.field(
            default="plane_parallel",
            converter=AtmosphereGeometry.convert,
            validator=attrs.validators.instance_of(
                (PlaneParallelGeometry, SphericalShellGeometry)
            ),
        ),
        doc="Problem geometry.",
        type=".PlaneParallelGeometry or .SphericalShellGeometry",
        init_type="str or dict or .AtmosphereGeometry",
        default='"plane_parallel"',
    )

    atmosphere: t.Optional[Atmosphere] = documented(
        attrs.field(
            factory=HomogeneousAtmosphere,
            converter=attrs.converters.optional(atmosphere_factory.convert),
            validator=attrs.validators.optional(
                attrs.validators.instance_of(Atmosphere)
            ),
        ),
        doc="Atmosphere specification. If set to ``None``, no atmosphere will "
        "be added. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.atmosphere_factory`.",
        type=":class:`.Atmosphere` or None",
        init_type=":class:`.Atmosphere` or dict or None",
        default=":class:`HomogeneousAtmosphere() <.HomogeneousAtmosphere>`",
    )

    surface: t.Optional[BasicSurface] = documented(
        attrs.field(
            factory=lambda: BasicSurface(bsdf=LambertianBSDF()),
            converter=attrs.converters.optional(_surface_converter),
            validator=attrs.validators.optional(
                attrs.validators.instance_of(BasicSurface)
            ),
        ),
        doc="Surface specification. If set to ``None``, no surface will be "
        "added. This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.surface_factory` and :data:`.bsdf_factory`.",
        type=".BasicSurface or None",
        init_type=".BasicSurface or .BSDF or dict, optional",
        default=":class:`BasicSurface(bsdf=LambertianBSDF()) <.BasicSurface>`",
    )

    dem: t.Optional[DEMSurface] = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(surface_factory.convert),
            validator=attrs.validators.optional(
                attrs.validators.instance_of(DEMSurface)
            ),
        ),
        doc="Digital elevation model (DEM) specification. If set to ``None``, no DEM will be "
        "added. This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.surface_factory`",
        type=".DEMSurface or None",
        init_type=".DEMSurface or dict, optional",
        default="None",
    )

    _integrator: Integrator = documented(
        attrs.field(
            factory=VolPathIntegrator,
            converter=integrator_factory.convert,
            validator=attrs.validators.instance_of(Integrator),
        ),
        doc=get_doc(Experiment, attrib="_integrator", field="doc"),
        type=get_doc(Experiment, attrib="_integrator", field="type"),
        init_type=get_doc(Experiment, attrib="_integrator", field="init_type"),
        default=":class:`VolPathIntegrator() <.VolPathIntegrator>`",
    )

    def __attrs_post_init__(self):
        self._normalize_measures()

    @property
    def _default_surface_width(self):
        return 1.0 * ureg.km

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        result = KernelDict({"type": "scene"})

        # Process atmosphere
        if self.atmosphere is not None:
            atmosphere = attrs.evolve(self.atmosphere, geometry=self.geometry)
            result.add(atmosphere, ctx=ctx)
        else:
            atmosphere = None

        # Process surface
        if self.surface is not None:
            if isinstance(self.geometry, PlaneParallelGeometry):
                if atmosphere is not None:
                    width = atmosphere.kernel_width_plane_parallel(ctx)
                    altitude = atmosphere.bottom
                else:
                    width = (
                        self.geometry.width
                        if self.geometry.width is not AUTO
                        else self._default_surface_width
                    )
                    altitude = 0.0 * ureg.km

                surface = attrs.evolve(
                    self.surface,
                    shape=RectangleShape.surface(altitude=altitude, width=width),
                )

            elif isinstance(self.geometry, SphericalShellGeometry):
                if atmosphere is not None:
                    altitude = self.atmosphere.bottom
                else:
                    altitude = 0.0 * ureg.km

                surface = attrs.evolve(
                    self.surface,
                    shape=SphereShape.surface(
                        altitude=altitude, planet_radius=self.geometry.planet_radius
                    ),
                )

            else:  # Shouldn't happen, prevented by validator
                raise RuntimeError

            result.add(surface, ctx=ctx)

        # Process DEM
        if self.dem is not None:
            for measure in self.measures:
                if isinstance(measure.target, TargetPoint):
                    warnings.warn(
                        UserWarning(
                            f"Your measure {measure.id}, uses a point target. "
                            f"This might be undesirable when simulating a DEM."
                        )
                    )

            result.add(self.dem, ctx=ctx)

        # Process measures
        for measure in self.measures:
            if measure_inside_atmosphere(atmosphere, measure, ctx):
                result.add(
                    measure,
                    ctx=ctx.evolve(atmosphere_medium_id=self.atmosphere.id_medium),
                )
            else:
                result.add(measure, ctx=ctx)

        # Process illumination
        result.add(self.illumination, ctx=ctx)

        # Process integrator
        result.add(self.integrator, ctx=ctx)

        return result

    def spectral_indices(self, measure: Measure) -> t.Generator[SpectralIndex]:
        """
        Generate spectral indices for a given measure.

        Parameters
        ----------
        measure : :class:`.Measure`
            Measure for which spectral indices are generated.
        
        Yields
        ------
        :class:`.SpectralIndex`
            Spectral index.

        Notes
        -----
        In monochromatic mode, the spectral indices are generated by the measure
        based on its spectral response function (``srf``) attribute.
        In CKD mode, the spectral indices are generated based on information
        from the molecular atmosphere's absorption dataset and from the measure's
        spectral response function (``srf``) attribute.
        """
        if eradiate.mode().is_mono:
            generator = self.spectral_indices_mono
        elif eradiate.mode().is_ckd:
            generator = self.spectral_indices_ckd
        else:
            raise RuntimeError(f"unsupported mode '{eradiate.mode().id}'")
        
        yield from generator(measure)

    def spectral_indices_mono(self, measure: Measure) -> t.Generator[MonoSpectralIndex]:
        yield from measure.spectral_indices()

    def spectral_indices_ckd(self, measure: Measure) -> t.Generator[CKDSpectralIndex]:
        if not self.atmosphere.is_molecular:
            yield from measure.spectral_indices()
        else:
            # assumes srf support is compact
            # InterpolatedSpectrum may not meet this assumption
            # TODO: handle non-compact support
            w_srf = np.stack([x.w for x in list(measure.spectral_indices())]).m_as("nm")
            ds = self.atmosphere.absorption_dataset
            bin_width = ureg(ds.attrs["bin_set"])
            w_ckd = np.unique(np.array(ds.bin.values, dtype=float))
            wmin, wmax = self.get_enclosing_w(w_ckd, w_srf)
            w_selected = w_ckd[(w_ckd >= wmin) & (w_ckd <= wmax)]
            w_selected *= ureg.nm

            ckd_bin_bounds = []
            for w in w_selected:
                ckd_bin_bounds.append((w - bin_width / 2, w + bin_width / 2))
            self._ckd_bin_bounds = ckd_bin_bounds

            for w in w_selected:
                for g in G16:
                    yield CKDSpectralIndex(w=w, g=g)

    @staticmethod
    def get_enclosing_w(w: np.ndarray, w_srf: np.ndarray) -> t.Tuple[float, float]:
        istart = np.argwhere(w_srf[0] < w)[0] - 1
        iend = np.argwhere(w_srf[-1] > w)[-1] + 1
        try:
            return w[istart], w[iend]
        except IndexError as e:
            raise ValueError("w_srf is not within w") from e

    def _normalize_measures(self) -> None:
        """
        Ensure that distant measure targets are set to appropriate values.
        Processed measures will have its ray target and origin parameters
        overridden if relevant.
        """
        for measure in self.measures:
            if isinstance(measure, DistantMeasure):
                if measure.target is None:
                    if isinstance(self.geometry, PlaneParallelGeometry):
                        # Plane parallel geometry: target ground level
                        target_point = [0.0, 0.0, 0.0] * ucc.get("length")

                    elif isinstance(self.geometry, SphericalShellGeometry):
                        # Spherical shell geometry: target ground level
                        target_point = [
                            0.0,
                            0.0,
                            self.geometry.planet_radius.m,
                        ] * self.geometry.planet_radius.units

                    else:  # Shouldn't happen, prevented by validator
                        raise RuntimeError

                    measure.target = TargetPoint(target_point)

    def _dataset_metadata(self, measure: Measure) -> t.Dict[str, str]:
        result = super(AtmosphereExperiment, self)._dataset_metadata(measure)

        if measure.is_distant():
            result["title"] = "Top-of-atmosphere simulation results"

        return result


__getattr__ = substitute(
    {
        "OneDimExperiment": (
            AtmosphereExperiment,
            {"deprecated_in": "0.22.5", "removed_in": "0.22.7"},
        )
    }
)
