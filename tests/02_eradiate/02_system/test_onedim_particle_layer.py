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
        surface=eradiate.scenes.bsdfs.LambertianBSDF(reflectance=r),
    )


def init_experiment_homogeneous_atmosphere(
    bottom, top, sigma_a, sigma_s, phase, r, w, spp
):
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
        surface=eradiate.scenes.bsdfs.LambertianBSDF(reflectance=r),
    )


@pytest.mark.parametrize(
    "w",
    [280, 400, 550, 650, 1000, 1500, 2400],
)
@pytest.mark.parametrize(
    "albedo",
    [0.2, 0.5, 0.8],
)
@pytest.mark.slow
def test(
    mode_mono_double,
    tmpdir,
    onedim_rayleigh_radprops,
    albedo,
    w,
    artefact_dir,
    ert_seed_state,
):
    r"""
    Equivalency of homogeneous atmosphere and corresponding particle layer
    ======================================================================

    Assert that a homogeneous atmosphere is equivalent to uniform particle layer
    parametrised such that the participating medium has the same radiative
    properties as the homogeneous atmosphere.

    Rationale
    ---------

    * Sensor: Distant measure covering a plane (11 angular points,
      10^6 sample per pixel) and targeting (0, 0, 0).
    * Illumination: Directional illumination with a zenith angle
      :math:`\theta = 30.0°` with black body irradiance spectrum.
    * Surface: a square surface with a Lambertian BRDF with 35 % reflectance.
    * Atmosphere:
      * in experiment 1: a 5 km high homogeneous atmosphere with
        a tabulated phase function corresponding to the Rayleigh phase function.
        The scattering coefficient is given by
        :meth:`eradiate.radprops.rayleigh.compute_sigma_s_air`.
      * in experiment 2: a uniform 5 km thick particle layer with
        a tabulated phase function corresponding to the Rayleigh phase function.
        The optical thickness at 550 nm of the particle layer is set to a value
        that match the optical thickness at 550 nm of the homogeneous
        atmosphere in experiment 1.
    * Integrator: volumetric path tracer.

    Expected behaviour
    ------------------

    Distant bidirectional reflectance factors measured by both experiments must
    agree within 0.5%.

    Results
    -------

    .. image:: generated/plots/test_onedim_particle_layer_280.0_0.2.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_280.0_0.5.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_280.0_0.8.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_400.0_0.2.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_400.0_0.5.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_400.0_0.8.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_550.0_0.2.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_550.0_0.5.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_550.0_0.8.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_650.0_0.2.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_650.0_0.5.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_650.0_0.8.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_1000.0_0.2.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_1000.0_0.5.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_1000.0_0.8.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_1500.0_0.2.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_1500.0_0.5.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_1500.0_0.8.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_2400.0_0.2.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_2400.0_0.5.png
       :width: 95%
    .. image:: generated/plots/test_onedim_particle_layer_2400.0_0.8.png
       :width: 95%
    """
    w = w * ureg.nm
    spp = 1e6
    reflectance = 1.0
    bottom = 0.0 * ureg.km
    top = 5.0 * ureg.km

    radprops = onedim_rayleigh_radprops(albedo=albedo)
    w_units = radprops.w.attrs["units"]
    sigma_t = to_quantity(radprops.sigma_t.interp(w=w.m_as(w_units)))
    albedo = to_quantity(radprops.albedo.interp(w=w.m_as(w_units)))
    sigma_s = sigma_t * albedo
    sigma_a = sigma_t - sigma_s
    phase = eradiate.scenes.phase.TabulatedPhaseFunction(data=radprops.phase)
    sigma_t_550 = to_quantity(
        radprops.sigma_t.interp(
            w=(550 * ureg.nm).m_as(w_units),
        )
    )
    height = top - bottom
    tau_550 = sigma_t_550 * height  # the layer is uniform
    dataset_path = tmpdir / "radprops.nc"
    radprops.to_netcdf(dataset_path)

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
    filename = f"test_onedim_particle_layer_{w.magnitude}_{albedo.m}.png"
    outdir = os.path.join(artefact_dir, "plots")
    os.makedirs(outdir, exist_ok=True)
    fname_plot = os.path.join(outdir, filename)
    make_figure(fname_plot=fname_plot, brf_1=brf_1, brf_2=brf_2)

    outcome = np.allclose(brf_1.values, brf_2.values, rtol=5e-3)

    if outcome is False:
        print(f"Test failed, see artefact {fname_plot}")
        assert False


@pytest.mark.parametrize(
    "w",
    [280, 400, 550, 650, 1000, 1500, 2400],
)
@pytest.mark.slow
def test_particle_layer_energy_conservation(
    mode_mono_double, tmpdir, onedim_rayleigh_radprops, w, artefact_dir, ert_seed_state
):
    r"""
    Energy  conservation in a non-absorbing particle layer
    ======================================================

    We assert that the reflected energy is equivalent to the irradiated energy in
    a scene with non-absorbing atmosphere and surface reflectivity of 1.

    Rationale
    ---------

    * Sensor: Distant flux measure covering the hemisphere (normal is (0, 0, 1),
      1e3 samples per pixel).
    * Illumination: Directional illumination with a zenith angle
      :math:`\theta = 30.0°` with black body irradiance spectrum.
    * Surface: a square surface with a Lambertian BRDF with 100 % reflectance.
    * Atmosphere:
      a non-absorbing uniform 5 km thick particle layer with
      a tabulated phase function corresponding to the Rayleigh phase function.
    * Integrator: volumetric path tracer.

    Expected behaviour
    ------------------

    The retrieved albedo must equal to 1.
    """
    w = w * ureg.nm
    spp = 1e3
    reflectance = 1.0
    bottom = 0.0 * ureg.km
    top = 5.0 * ureg.km

    radprops = onedim_rayleigh_radprops(albedo=1.0)

    w_units = radprops.w.attrs["units"]
    sigma_t_550 = to_quantity(
        radprops.sigma_t.interp(
            w=(550 * ureg.nm).m_as(w_units),
        )
    )
    height = top - bottom
    tau_550 = sigma_t_550 * height  # the layer is uniform
    dataset_path = tmpdir / "radprops.nc"
    radprops.to_netcdf(dataset_path)

    experiment = eradiate.experiments.OneDimExperiment(
        measures=[
            eradiate.scenes.measure.DistantFluxMeasure(
                film_resolution=(32, 32),
                target=eradiate.scenes.measure.TargetRectangle(
                    xmin=-20.0 * ureg.km,
                    xmax=20.0 * ureg.km,
                    ymin=-20.0 * ureg.km,
                    ymax=20.0 * ureg.km,
                    z=5.0 * ureg.km,
                ),
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
        surface=eradiate.scenes.bsdfs.LambertianBSDF(reflectance=reflectance),
    )

    ert_seed_state.reset()
    experiment.run(seed_state=ert_seed_state)

    albedo = experiment.results["measure"]["albedo"]

    # Make figure
    outdir = os.path.join(artefact_dir, "plots")
    os.makedirs(outdir, exist_ok=True)

    outcome = np.allclose(albedo.values, 1.0, rtol=5e-3)

    if outcome is False:
        print(f"Test failed, expected albedo = 1.0, got {albedo.values}")
        assert False
