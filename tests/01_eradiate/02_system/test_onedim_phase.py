"""Test cases targeting the phase function component of a 1D experiment."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.experiments import AtmosphereExperiment
from eradiate.units import to_quantity


def make_figure(fname_plot, brf_1, brf_2):
    fig = plt.figure(figsize=(8, 5))
    params = dict(x="vza", ls="dotted", marker=".")
    brf_1.plot(**params)
    brf_2.plot(**params)
    plt.legend(["experiment 1", "experiment 2"])
    plt.title(f"w = {to_quantity(brf_1.w).squeeze():~}")
    plt.tight_layout()

    print(f"Saving figure to {fname_plot}")
    fig.savefig(fname_plot, dpi=200)
    plt.close()


@pytest.mark.slow
def test(mode_mono_double, rayleigh_tab_phase, artefact_dir, ert_seed_state, request):
    r"""
    Equivalency of plugin and tabulated versions of Rayleigh phase function
    =======================================================================

    Assert that a homogeneous atmosphere with a Rayleigh phase function is
    equivalent to a homogeneous atmosphere with a tabulated version of the
    Rayleigh phase function.

    Rationale
    ---------

    * Sensor: Distant measure covering a plane (11 angular points,
      10^6 sample per pixel) and targeting (0, 0, 0).
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

    Distant bidirectional reflectance factors measured by both experiments must
    agree within 0.5%.

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
    wavelengths = [280, 400, 550, 650, 1000, 1500, 2400] * ureg.nm
    spp = int(1e6)
    reflectance = 0.35
    bottom = 0.0 * ureg.km
    top = 10.0 * ureg.km

    def init_experiment(sigma_a, sigma_s, phase):
        assert eradiate.mode().id == "mono_double"

        return AtmosphereExperiment(
            geometry={
                "type": "plane_parallel",
                "ground_altitude": bottom,
                "toa_altitude": top,
            },
            measures={
                "type": "mdistant",
                "construct": "hplane",
                "srf": {"type": "multi_delta", "wavelengths": wavelengths},
                "zeniths": np.linspace(-75, 75, 11) * ureg.deg,
                "azimuth": 0.0 * ureg.deg,
            },
            illumination={
                "type": "directional",
                "zenith": 30 * ureg.deg,
                "azimuth": 0.0 * ureg.deg,
                "irradiance": {"type": "solar_irradiance", "dataset": "blackbody_sun"},
            },
            atmosphere={
                "type": "homogeneous",
                "sigma_a": sigma_a,
                "sigma_s": sigma_s,
                "phase": phase,
            },
            surface={"type": "lambertian", "reflectance": reflectance},
        )

    experiment_1 = init_experiment(
        sigma_a=0.0,
        sigma_s=eradiate.scenes.spectra.AirScatteringCoefficientSpectrum(),
        phase=eradiate.scenes.phase.RayleighPhaseFunction(),
    )

    ert_seed_state.reset()
    eradiate.run(experiment_1, seed_state=ert_seed_state, spp=spp)

    radprops = rayleigh_tab_phase(albedo=1.0)
    w = to_quantity(radprops.w)
    sigma_t = to_quantity(radprops.sigma_t)
    albedo = to_quantity(radprops.albedo)
    sigma_s_2 = sigma_t * albedo
    sigma_a_2 = sigma_t * (1.0 - albedo)
    phase_2 = eradiate.scenes.phase.TabulatedPhaseFunction(data=radprops.phase)

    experiment_2 = init_experiment(
        sigma_a={
            "type": "interpolated",
            "values": sigma_a_2,
            "wavelengths": w,
        },
        sigma_s={
            "type": "interpolated",
            "values": sigma_s_2,
            "wavelengths": w,
        },
        phase=phase_2,
    )

    ert_seed_state.reset()
    eradiate.run(experiment_2, seed_state=ert_seed_state, spp=spp)

    brf_1 = experiment_1.results["measure"]["brf"]
    brf_2 = experiment_2.results["measure"]["brf"]

    # Make figure
    for w in wavelengths.magnitude:
        filename = f"{request.node.originalname}_{w}.png"
        outdir = os.path.join(artefact_dir, "plots")
        os.makedirs(outdir, exist_ok=True)
        fname_plot = os.path.join(outdir, filename)
        make_figure(fname_plot=fname_plot, brf_1=brf_1.sel(w=w), brf_2=brf_2.sel(w=w))

    np.testing.assert_allclose(brf_1.values, brf_2.values, rtol=5e-3)
