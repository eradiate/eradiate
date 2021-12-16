"""Test cases targeting the phase function component of a 1D experiment."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.units import to_quantity


def init_experiment(bottom, top, sigma_a, sigma_s, phase, r, w, spp):
    assert eradiate.mode().id == "mono_double"

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


@pytest.mark.parametrize(
    "w",
    [280, 400, 550, 650, 1000, 1500, 2400],
)
@pytest.mark.slow
def test(mode_mono_double, onedim_rayleigh_radprops, w, artefact_dir):
    r"""
    Equivalency of plugin and tabulated versions of Rayleigh phase function
    =======================================================================

    Assert that a homogeneous atmosphere with a Rayleigh phase function is
    equivalent to a homogeneous atmosphere with a tabulated version of the
    Rayleigh phase function.

    Rationale
    ---------

    * Sensor: Distant measure covering a plane (11 angular points,
      1000 sample per pixel) and targeting (0, 0, 0).
    * Illumination: Directional illumination with a zenith angle
      :math:`\theta = 30.0°` with black body irradiance spectrum.
    * Surface: a square surface with a Lambertian BRDF with 35 % reflectance.
    * Atmosphere: a non-absorbing 10 km high homogeneous atmosphere. The
      scattering phase function is defined in two ways in two experiments:
      * in experiment 1: using
        :class:`~eradiate.scenes.phase.RayleighPhaseFunction`
      * in experiment 2: using
        :class:`~eradiate.scenes.phase.TabulatedPhaseFunction` with Rayleigh
        phase function tabulated values.
    * Integrator: volumetric path tracer.

    Expected behaviour
    ------------------

    Distant bi-directional reflectance factors measured by both experiments must
    agree within 0.01%, outliers excluded.

    Results
    -------

    .. image:: generated/plots/test_onedim_phase_300.0.png
       :width: 95%
    .. image:: generated/plots/test_onedim_phase_550.0.png
       :width: 95%
    .. image:: generated/plots/test_onedim_phase_650.0.png
       :width: 95%
    .. image:: generated/plots/test_onedim_phase_850.0.png
       :width: 95%
    .. image:: generated/plots/test_onedim_phase_1000.0.png
       :width: 95%
    .. image:: generated/plots/test_onedim_phase_1500.0.png
       :width: 95%
    .. image:: generated/plots/test_onedim_phase_2000.0.png
       :width: 95%
    .. image:: generated/plots/test_onedim_phase_2400.0.png
       :width: 95%
    """
    w = w * ureg.nm
    spp = 1e3
    reflectance = 0.35
    bottom = 0.0 * ureg.km
    top = 10.0 * ureg.km

    sigma_a_1 = 0.0
    sigma_s_1 = eradiate.scenes.spectra.AirScatteringCoefficientSpectrum()
    phase_1 = eradiate.scenes.phase.RayleighPhaseFunction()

    w_units = onedim_rayleigh_radprops.w.attrs["units"]
    sigma_t = to_quantity(onedim_rayleigh_radprops.sigma_t.interp(w=w.m_as(w_units)))
    albedo = to_quantity(onedim_rayleigh_radprops.albedo.interp(w=w.m_as(w_units)))
    sigma_s_2 = sigma_t * albedo
    sigma_a_2 = sigma_t * (1.0 - albedo)
    phase_2 = eradiate.scenes.phase.TabulatedPhaseFunction(
        data=onedim_rayleigh_radprops.phase
    )

    experiment_1 = init_experiment(
        bottom=bottom,
        top=top,
        sigma_a=sigma_a_1,
        sigma_s=sigma_s_1,
        phase=phase_1,
        r=reflectance,
        w=w,
        spp=spp,
    )

    experiment_2 = init_experiment(
        bottom=bottom,
        top=top,
        sigma_a=sigma_a_2,
        sigma_s=sigma_s_2,
        phase=phase_2,
        r=reflectance,
        w=w,
        spp=spp,
    )

    experiment_1.run()
    experiment_2.run()

    brf_1 = experiment_1.results["measure"]["brf"]
    brf_2 = experiment_2.results["measure"]["brf"]

    # Make figure
    filename = f"test_onedim_phase_{w.magnitude}.png"
    outdir = os.path.join(artefact_dir, "plots")
    os.makedirs(outdir, exist_ok=True)
    fname_plot = os.path.join(outdir, filename)
    make_figure(fname_plot=fname_plot, brf_1=brf_1, brf_2=brf_2)

    outcome = np.allclose(brf_1.values, brf_2.values, rtol=1e-4)

    if outcome is False:
        print(f"Test failed, see artefact {fname_plot}")
        assert False
