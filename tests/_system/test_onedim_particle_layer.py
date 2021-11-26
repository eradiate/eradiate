"""Test cases targetting the particle layer component of a 1D experiment."""
import os

import numpy as np
import pytest
import seaborn

import eradiate
from eradiate import unit_registry as ureg
from eradiate.units import to_quantity

from .test_onedim_phase import ensure_output_dir, make_figure, rayleigh_radprops

seaborn.set_style("ticks")

eradiate_dir = os.environ["ERADIATE_DIR"]
output_dir = os.path.join(eradiate_dir, "test_report", "generated")


def init_experiment_particle_layer(bottom, top, dataset_path, tau_550, r, w, spp):
    eradiate.set_mode("mono_double")

    return eradiate.experiments.OneDimExperiment(
        measures=[
            eradiate.scenes.measure.MultiDistantMeasure.from_viewing_angles(
                spectral_cfg=eradiate.scenes.measure.MeasureSpectralConfig.new(
                    wavelengths=w,
                ),
                zeniths=np.linspace(-75, 75, 11) * ureg.deg,
                azimuths=0.0 * ureg.deg,
                spp=spp,
            )
        ],
        illumination=eradiate.scenes.illumination.DirectionalIllumination(
            zenith=30 * ureg.deg,
            azimuth=0.0 * ureg.deg,
            irradiance=eradiate.scenes.spectra.SolarIrradianceSpectrum(
                dataset="blackbody_sun"
            ),
        ),
        atmosphere=eradiate.scenes.atmosphere.HeterogeneousAtmosphere(
            molecular_atmosphere=None,
            particle_layers=[
                eradiate.scenes.atmosphere.ParticleLayer(
                    bottom=bottom,
                    top=top,
                    dataset=dataset_path,
                    tau_550=tau_550,
                )
            ],
        ),
        surface=eradiate.scenes.surface.LambertianSurface(reflectance=r),
    )


def init_experiment_homogeneous_atmosphere(
    bottom, top, sigma_a, sigma_s, phase, r, w, spp
):
    eradiate.set_mode("mono_double")

    return eradiate.experiments.OneDimExperiment(
        measures=[
            eradiate.scenes.measure.MultiDistantMeasure.from_viewing_angles(
                spectral_cfg=eradiate.scenes.measure.MeasureSpectralConfig.new(
                    wavelengths=w,
                ),
                zeniths=np.linspace(-75, 75, 11) * ureg.deg,
                azimuths=0.0 * ureg.deg,
                spp=spp,
            )
        ],
        illumination=eradiate.scenes.illumination.DirectionalIllumination(
            zenith=30 * ureg.deg,
            azimuth=0.0 * ureg.deg,
            irradiance=eradiate.scenes.spectra.SolarIrradianceSpectrum(
                dataset="blackbody_sun"
            ),
        ),
        atmosphere=eradiate.scenes.atmosphere.HomogeneousAtmosphere(
            bottom=bottom,
            top=top,
            sigma_a=sigma_a,
            sigma_s=sigma_s,
            phase=phase,
        ),
        surface=eradiate.scenes.surface.LambertianSurface(reflectance=r),
    )


@pytest.mark.parametrize(
    "w",
    np.array([280.0, 400.0, 550.0, 650.0, 1000.0, 1500.0, 2400.0]) * ureg.nm,
)
@pytest.mark.slow
def test(tmpdir, rayleigh_radprops, w):
    r"""
    Equivalency of homogeneous atmosphere and corresponding particle layer
    ======================================================================

    Assert that a homogeneous atmosphere is equivalent to uniform particle layer
    parametrised such that the participating medium has the same radiative
    properties as the homogeneous atmosphere.

    Rationale
    ---------

    * Sensor: Distant measure covering a plane (11 angular points,
      1000 sample per pixel) and targeting (0, 0, 0).
    * Illumination: Directional illumination with a zenith angle
      :math:`\theta = 30.0°` with black body irradiance spectrum.
    * Surface: a square surface with a Lambertian BRDF with 35 % reflectance.
    * Atmosphere:
      * in experiment 1: a non-absorbing 5 km high homogeneous atmosphere with
        a tabulated phase function corresponding to the Rayleigh phase function.
        The scattering coefficient is given by
        :meth:`eradiate.radprops.rayleigh.compute_sigma_s_air`.
      * in experiment 2: a non-absorbing uniform 5 km thick particle layer with
        a tabulated phase function corresponding to the Rayleigh phase function.
        The optical thickness at 550 nm of the particle layer is set to a value
        that match the optical thickness at 550 nm of the homogeneous
        atmosphere in experiment 1.
    * Integrator: volumetric path tracer.

    Expected behaviour
    ------------------

    Distant bi-directional reflectance factors measured by both experiments must
    agree within 1%, outliers excluded.

    Results
    -------

    .. image:: generated/plots/test_onedim_particle_layer_280.0.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_400.0.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_550.0.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_650.0.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_1000.0.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_1500.0.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_2400.0.png
       :width: 95%
    """
    spp = 1e3
    reflectance = 0.35
    bottom = 0.0 * ureg.km
    top = 5.0 * ureg.km

    w_units = rayleigh_radprops.w.attrs["units"]
    sigma_t = to_quantity(rayleigh_radprops.sigma_t.interp(w=w.m_as(w_units)))
    albedo = to_quantity(rayleigh_radprops.albedo.interp(w=w.m_as(w_units)))
    sigma_s = sigma_t * albedo
    sigma_a = sigma_t * (1.0 - albedo)
    phase = eradiate.scenes.phase.TabulatedPhaseFunction(data=rayleigh_radprops.phase)

    sigma_s_550 = eradiate.radprops.rayleigh.compute_sigma_s_air(
        wavelength=550.0 * ureg.nm
    )
    height = top - bottom
    tau_550 = sigma_s_550 * height

    dataset_path = tmpdir / "radprops.nc"
    rayleigh_radprops.to_netcdf(dataset_path)

    experiment_1 = init_experiment_particle_layer(
        bottom=bottom,
        top=top,
        dataset_path=dataset_path,
        tau_550=tau_550,
        r=reflectance,
        w=w,
        spp=spp,
    )

    experiment_2 = init_experiment_homogeneous_atmosphere(
        bottom=bottom,
        top=top,
        sigma_a=sigma_a,
        sigma_s=sigma_s,
        phase=phase,
        r=reflectance,
        w=w,
        spp=spp,
    )

    with eradiate.unit_context_kernel.override(length="km"):
        experiment_1.run()
        experiment_2.run()

    brf_1 = experiment_1.results["measure"]["brf"]
    brf_2 = experiment_2.results["measure"]["brf"]

    # Make figure
    filename = f"test_onedim_particle_layer_{w.magnitude}.png"
    ensure_output_dir(os.path.join(output_dir, "plots"))
    fname_plot = os.path.join(output_dir, "plots", filename)
    make_figure(fname_plot=fname_plot, brf_1=brf_1, brf_2=brf_2)

    # exclude outliers that are due to the batman issue
    outliers_1 = np.isclose(brf_1.values, reflectance, rtol=1e-5)
    outliers_2 = np.isclose(brf_2.values, reflectance, rtol=1e-5)
    no_outliers = ~outliers_1 & ~outliers_2

    assert np.allclose(
        brf_1.where(no_outliers).values,
        brf_2.where(no_outliers).values,
        rtol=1e-2,
        equal_nan=True,
    )
