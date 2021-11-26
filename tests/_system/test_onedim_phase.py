"""Test cases targeting the phase function component of a 1D experiment."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn
import xarray as xr

import eradiate
from eradiate import unit_registry as ureg
from eradiate.units import to_quantity

seaborn.set_style("ticks")

eradiate_dir = os.environ["ERADIATE_DIR"]
output_dir = os.path.join(eradiate_dir, "test_report", "generated")


@pytest.fixture
def rayleigh_radprops():
    """
    Radiative properties of a Rayleigh scattering medium in [280, 2400] nm.
    """
    w = np.linspace(279.0, 2401.0, 10000) * ureg.nm

    # Collision coefficients
    sigma_s = eradiate.radprops.rayleigh.compute_sigma_s_air(wavelength=w)
    albedo = np.ones_like(sigma_s) * ureg.dimensionless  # no absorption
    sigma_t = sigma_s * albedo

    # Phase function
    # Note: rayleigh phase function does not change with wavelength
    def rayleigh_phase_function(mu):
        magnitude = 3.0 * (1 + np.square(mu)) / (16 * np.pi)
        return magnitude / ureg.steradian

    mu = np.linspace(-1.0, 1.0)
    arrays = [rayleigh_phase_function(mu) for _ in w]
    phase = np.stack(arrays, axis=0).reshape(w.size, mu.size, 1, 1)

    return xr.Dataset(
        data_vars={
            "sigma_t": (
                "w",
                sigma_t.magnitude,
                dict(
                    standard_name="air_volume_extinction_coefficient",
                    units=f"{sigma_t.units:~}",
                ),
            ),
            "albedo": (
                "w",
                albedo.magnitude,
                dict(
                    standard_name="single_scattering_albedo", units=f"{albedo.units:~}"
                ),
            ),
            "phase": (
                ("w", "mu", "i", "j"),
                phase.magnitude,
                dict(standard_name="scattering_phase_matrix", units=f"{phase.units:~}"),
            ),
        },
        coords={
            "w": ("w", w.magnitude, dict(units=f"{w.units:~}")),
            "mu": (
                "mu",
                mu,
                dict(standard_name="scattering_angle_cosine", units="dimensionless"),
            ),
            "i": ("i", [0]),
            "j": ("j", [0]),
        },
    )


def init_experiment(bottom, top, sigma_a, sigma_s, phase, r, w, spp):
    assert eradiate.mode().id == "mono"

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


def ensure_output_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def make_figure(fname_plot, brf_1, brf_2):
    fig = plt.figure(figsize=(8, 5))
    params = dict(x="vza", ls="dotted", marker=".")
    brf_1.plot(**params)
    brf_2.plot(**params)
    plt.legend(["experiment 1", "experiment 2"])
    plt.tight_layout()
    fig.savefig(fname_plot, dpi=200)
    plt.close()


@pytest.mark.parametrize(
    "w",
    np.array([280.0, 400.0, 550.0, 650.0, 1000.0, 1500.0, 2400.0]) * ureg.nm,
)
@pytest.mark.slow
def test(mode_mono, rayleigh_radprops, w):
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
    agree within 1%, outliers excluded.

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
    spp = 1e3
    reflectance = 0.35
    bottom = 0.0 * ureg.km
    top = 10.0 * ureg.km

    sigma_a_1 = 0.0
    sigma_s_1 = eradiate.scenes.spectra.AirScatteringCoefficientSpectrum()
    phase_1 = eradiate.scenes.phase.RayleighPhaseFunction()

    w_units = rayleigh_radprops.w.attrs["units"]
    sigma_t = to_quantity(rayleigh_radprops.sigma_t.interp(w=w.m_as(w_units)))
    albedo = to_quantity(rayleigh_radprops.albedo.interp(w=w.m_as(w_units)))
    sigma_s_2 = sigma_t * albedo
    sigma_a_2 = sigma_t * (1.0 - albedo)
    phase_2 = eradiate.scenes.phase.TabulatedPhaseFunction(data=rayleigh_radprops.phase)

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

    with eradiate.unit_context_kernel.override(length="km"):
        experiment_1.run()

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

    with eradiate.unit_context_kernel.override(length="km"):
        experiment_2.run()

    brf_1 = experiment_1.results["measure"]["brf"]
    brf_2 = experiment_2.results["measure"]["brf"]

    # Make figure
    filename = f"test_onedim_phase_{w.magnitude}.png"
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
