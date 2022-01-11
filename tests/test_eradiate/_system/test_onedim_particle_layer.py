"""Test cases targetting the particle layer component of a 1D experiment."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.units import to_quantity


def make_figure(fname_plot, brf_1, brf_2):
    fig = plt.figure(figsize=(8, 5))
    params = dict(x="vza", ls="dotted", marker=".")
    brf_1.plot(**params)
    brf_2.plot(**params)
    plt.legend(["experiment 1", "experiment 2"])
    plt.title(f"w = {to_quantity(brf_1.w).squeeze():~}")
    plt.tight_layout()
    fig.savefig(fname_plot, dpi=200)
    plt.close()


def init_experiment_particle_layer(bottom, top, dataset_path, tau_550, r, w, spp):
    eradiate.set_mode("mono_double")

    return eradiate.experiments.OneDimExperiment(
        measures=[
            eradiate.scenes.measure.MultiDistantMeasure.from_viewing_angles(
                spectral_cfg=eradiate.scenes.measure.MeasureSpectralConfig.new(
                    wavelengths=w,
                ),
                zeniths=np.linspace(-75, 75, 51) * ureg.deg,
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
                zeniths=np.linspace(-75, 75, 51) * ureg.deg,
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
    [280, 400, 550, 650, 1000, 1500, 2400],
)
@pytest.mark.slow
def test(tmpdir, onedim_rayleigh_radprops, w, artefact_dir, ert_seed_state):
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
    w = w * ureg.nm
    spp = 1e3
    reflectance = 0.35
    bottom = 0.0 * ureg.km
    top = 5.0 * ureg.km

    w_units = onedim_rayleigh_radprops.w.attrs["units"]
    sigma_t = to_quantity(onedim_rayleigh_radprops.sigma_t.interp(w=w.m_as(w_units)))
    albedo = to_quantity(onedim_rayleigh_radprops.albedo.interp(w=w.m_as(w_units)))
    sigma_s = sigma_t * albedo
    sigma_a = sigma_t * (1.0 - albedo)
    phase = eradiate.scenes.phase.TabulatedPhaseFunction(
        data=onedim_rayleigh_radprops.phase
    )

    sigma_s_550 = eradiate.radprops.rayleigh.compute_sigma_s_air(
        wavelength=550.0 * ureg.nm
    )
    height = top - bottom
    tau_550 = sigma_s_550 * height

    dataset_path = tmpdir / "radprops.nc"
    onedim_rayleigh_radprops.to_netcdf(dataset_path)

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

    ert_seed_state.reset()
    experiment_1.run(seed_state=ert_seed_state)

    ert_seed_state.reset()
    experiment_2.run(seed_state=ert_seed_state)

    brf_1 = experiment_1.results["measure"]["brf"]
    brf_2 = experiment_2.results["measure"]["brf"]

    # Make figure
    filename = f"test_onedim_particle_layer_{w.magnitude}.png"
    outdir = os.path.join(artefact_dir, "plots")
    os.makedirs(outdir, exist_ok=True)
    fname_plot = os.path.join(outdir, filename)
    make_figure(fname_plot=fname_plot, brf_1=brf_1, brf_2=brf_2)

    # exclude outliers that are due to the batman issue
    outcome = np.allclose(brf_1.values, brf_2.values, atol=reflectance * 1e-3)

    if outcome is False:
        print(f"Test failed, see artefact {fname_plot}")
        assert False
