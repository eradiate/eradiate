"""Test cases asserting the results consistency between the mono and ckd modes."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg, unit_context_kernel as uck

eradiate_dir = os.environ["ERADIATE_DIR"]
output_dir = os.path.join(eradiate_dir, "test_report", "generated")


def ensure_output_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def init_mono_experiment(wavelengths, spp, reflectance, zeniths):
    """
    Initialise the monochromatic experiment.
    """
    eradiate.set_mode("mono_double")

    return eradiate.experiments.OneDimExperiment(
        measures=[
            eradiate.scenes.measure.MultiDistantMeasure.from_viewing_angles(
                spectral_cfg=eradiate.scenes.measure.MeasureSpectralConfig.new(
                    wavelengths=wavelengths
                ),
                zeniths=zeniths,
                azimuths=0.0 * ureg.deg,
                spp=spp,
            )
        ],
        illumination=eradiate.scenes.illumination.DirectionalIllumination(
            zenith=50.0 * ureg.deg, azimuth=0.0 * ureg.deg
        ),
        atmosphere=eradiate.scenes.atmosphere.MolecularAtmosphere.afgl_1986(
            id="molecules",
            has_absorption=True,
            has_scattering=True,
            absorption_data_sets={},
        ),
        surface=eradiate.scenes.surface.LambertianSurface(
            reflectance=reflectance,
        ),
    )


def init_ckd_experiment(bin_set, bins, spp, reflectance, zeniths):
    """
    Initialise CKD experiment.
    """
    eradiate.set_mode("ckd_double")

    return eradiate.experiments.OneDimExperiment(
        measures=[
            eradiate.scenes.measure.MultiDistantMeasure.from_viewing_angles(
                spectral_cfg=eradiate.scenes.measure.MeasureSpectralConfig.new(
                    bin_set=bin_set,
                    bins=bins,
                ),
                zeniths=zeniths,
                azimuths=0.0 * ureg.deg,
                spp=spp,
            )
        ],
        illumination=eradiate.scenes.illumination.DirectionalIllumination(
            zenith=50.0 * ureg.deg, azimuth=0.0 * ureg.deg
        ),
        atmosphere=eradiate.scenes.atmosphere.MolecularAtmosphere.afgl_1986(
            id="molecules",
            has_absorption=True,
            has_scattering=True,
        ),
        surface=eradiate.scenes.surface.LambertianSurface(
            reflectance=reflectance,
        ),
    )


def make_figure(
    ckd_results, mono_results_averaged, fname_plot, wavelength_bin, reflectance
):
    fig = plt.figure(figsize=(8, 5))
    brf_mono_averaged = mono_results_averaged.brf.squeeze()
    brf_ckd = ckd_results.brf.squeeze()
    brf_mono_averaged.plot(x="vza", ls="dotted", marker=".")
    brf_ckd.plot(x="vza", ls="dotted", marker=".")

    # Compute relative absolute difference and annotate plot with min, max and
    # mean values
    # NB: RAD(x, y) = abs(x - y) / x
    rad = np.divide(
        np.abs(brf_mono_averaged.values - brf_ckd.values),
        brf_mono_averaged.values,
        where=brf_mono_averaged.values != 0.0,
        out=np.full_like(brf_mono_averaged.values, fill_value=np.nan),
    )
    rmad = np.mean(rad[~np.isnan(rad)])
    minrad = np.min(rad[~np.isnan(rad)])
    maxrad = np.max(rad[~np.isnan(rad)])
    text = (
        f"mean(RAD) >= {round(100 * rmad, 2)} %\n"
        f"min(RAD) >= {round(100 * minrad, 2)} %\n"
        f"max(RAD) >= {round(100 * maxrad, 2)} %"
    )
    plt.text(
        0.5,
        0.9,
        text,
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
    )

    plt.legend(["mono_double (averaged)", "ckd_double"])
    plt.title(f"Wavelength bin: {wavelength_bin}, reflectance = {reflectance}")
    plt.tight_layout()
    fig.savefig(fname_plot, dpi=200)
    plt.close()


@pytest.mark.parametrize("reflectance", [0.0, 0.5, 1.0])
@pytest.mark.skipif_data_not_found("absorption_spectrum", "H2O-spectra-18000_18100")
@pytest.mark.skipif_data_not_found("absorption_spectrum", "H2O-spectra-18100_18200")
@pytest.mark.skipif_data_not_found("absorption_spectrum", "H2O-spectra-18200_18300")
@pytest.mark.skipif_data_not_found("absorption_spectrum", "H2O-spectra-18300_18400")
@pytest.mark.slow
def test_550(reflectance):
    r"""
    Results consistency between `mono_double` and `ckd_double` modes
    ================================================================

    This test checks the results consistency between mono_double and ckd_double
    modes in the [545, 555] nm wavelength bin.

    Rationale
    ---------

    * Sensor: Distant measure covering a plane (11 angular points,
      10000 sample per pixel) and targeting (0, 0, 0).
    * Illumination: Directional illumination with a zenith angle
      :math:`\theta = 50.0°`.
    * Surface: a square surface with a Lambertian BRDF with reflectance
      parameter in :math:`\rho \in [0.0, 0.5, 1.0]`.
    * Atmosphere: a molecular atmosphere derived from the AFGL (1986)
      thermophysical properties `us_standard` model, with both scattering and
      absorption enabled.
    * Integrator: volumetric path tracer.

    Expected behaviour
    ------------------

    The [545, 555] nm - integrated BRF results obtained with the `mono_double`
    and `ckd_double` modes should agree within 5%.

    The `mono_double` BRF results are spectrally averaged according to the
    following formula:

    .. math::

       \hat{q} = \frac{
          \int_{
             \lambda_{\mathrm{min}}
          }^{
             \lambda_{\mathrm{max}}
          }
          q(\lambda) \, \mathrm{d} \lambda
       }{
          \int_{
             \lambda_{\mathrm{min}}
          }^{
             \lambda_{\mathrm{max}}
          }
          \mathrm{d} \lambda
       }

    where

    * :math:`q` is some spectral quantity (e.g. the BRF)
    * :math:`\lambda` denotes the wavelength
    * :math:`[\lambda_{\mathrm{min}},\, \lambda_{\mathrm{max}}]` is the
      wavelength bin

    Results
    -------

    Note: :math:`\mathrm{RAD}(x, y) = \left| \dfrac{x - y}{x} \right|`

    .. image:: generated/plots/test_onedim_ckd_vs_mono_550_0.0.png
       :width: 95%
    .. image:: generated/plots/test_onedim_ckd_vs_mono_550_0.5.png
       :width: 95%
    .. image:: generated/plots/test_onedim_ckd_vs_mono_550_1.0.png
       :width: 95%
    """
    # Settings
    zeniths = np.linspace(-75, 75, 11)
    spp = 1e4
    wavelengths = np.linspace(545.0, 555.0, 1001) * ureg.nm  # mono mode setting
    bin_set = "10nm"  # ckd mode setting
    bins = "550"  # ckd mode setting

    # Monochromatic experiment
    mono_exp = init_mono_experiment(
        wavelengths=wavelengths,
        spp=spp,
        reflectance=reflectance,
        zeniths=zeniths,
    )
    with uck.override(length="km"):  # this is a temporary workaround the 'batman' issue
        mono_exp.run()
    mono_results = mono_exp.results["measure"]

    wavelength_bin_width = (wavelengths[-1] - wavelengths[0]).m_as(mono_results.w.units)
    mono_results_integrated = mono_results.integrate(coord="w") / wavelength_bin_width

    # CKD experiment
    ckd_exp = init_ckd_experiment(
        bin_set=bin_set,
        bins=bins,
        spp=spp,
        reflectance=reflectance,
        zeniths=zeniths,
    )
    ckd_exp.run()
    ckd_results = ckd_exp.results["measure"]

    # Make figure
    filename = f"test_onedim_ckd_vs_mono_{bins}_{reflectance}.png"
    ensure_output_dir(os.path.join(output_dir, "plots"))
    fname_plot = os.path.join(output_dir, "plots", filename)
    make_figure(
        ckd_results=ckd_results,
        mono_results_averaged=mono_results_integrated,
        fname_plot=fname_plot,
        wavelength_bin="[545, 555] nm",
        reflectance=reflectance,
    )

    # Assert averaged mono results are consistent with ckd results
    mono_brf = mono_results_integrated.brf.squeeze().values
    ckd_brf = ckd_results.brf.squeeze().values
    assert np.allclose(mono_brf, ckd_brf, rtol=5e-2)


@pytest.mark.parametrize("reflectance", [0.0, 0.5, 1.0])
@pytest.mark.skipif_data_not_found("absorption_spectrum", "H2O-spectra-9400_9500")
@pytest.mark.skipif_data_not_found("absorption_spectrum", "H2O-spectra-9500_9600")
@pytest.mark.skipif_data_not_found("absorption_spectrum", "CO2-spectra-9400_9500")
@pytest.mark.skipif_data_not_found("absorption_spectrum", "CO2-spectra-9500_9600")
@pytest.mark.skipif_data_not_found("absorption_spectrum", "N2O-spectra-9400_9500")
@pytest.mark.skipif_data_not_found("absorption_spectrum", "N2O-spectra-9500_9600")
@pytest.mark.skipif_data_not_found("absorption_spectrum", "CO-spectra-9400_9500")
@pytest.mark.skipif_data_not_found("absorption_spectrum", "CO-spectra-9500_9600")
@pytest.mark.skipif_data_not_found("absorption_spectrum", "CH4-spectra-9400_9500")
@pytest.mark.skipif_data_not_found("absorption_spectrum", "CH4-spectra-9500_9600")
@pytest.mark.skipif_data_not_found("absorption_spectrum", "O2-spectra-9400_9500")
@pytest.mark.skipif_data_not_found("absorption_spectrum", "O2-spectra-9500_9600")
@pytest.mark.slow
def test_1050(reflectance):
    r"""
    Results consistency between `mono_double` and `ckd_double` modes
    ================================================================

    This test checks the results consistency between mono_double and ckd_double
    modes in the [1045, 1055] nm wavelength bin.

    Rationale
    ---------

    * Sensor: Distant measure covering a plane (11 angular points,
      10000 sample per pixel) and targeting (0, 0, 0).
    * Illumination: Directional illumination with a zenith angle
      :math:`\theta = 50.0°`.
    * Surface: a square surface with a Lambertian BRDF with reflectance
      parameter in :math:`\rho \in [0.0, 0.5, 1.0]`.
    * Atmosphere: a molecular atmosphere derived from the AFGL (1986)
      thermophysical properties `us_standard` model, with both scattering and
      absorption enabled.
    * Integrator: volumetric path tracer.

    Expected behaviour
    ------------------

    The [1045, 1055] nm - integrated BRF results obtained with the `mono_double`
    and `ckd_double` modes should agree within 5%.

    The `mono_double` BRF results are spectrally averaged according to the
    following formula:

    .. math::

       \hat{q} = \frac{
          \int_{
             \lambda_{\mathrm{min}}
          }^{
             \lambda_{\mathrm{max}}
          }
          q(\lambda) \, \mathrm{d} \lambda
       }{
          \int_{
             \lambda_{\mathrm{min}}
          }^{
             \lambda_{\mathrm{max}}
          }
          \mathrm{d} \lambda
       }

    where

    * :math:`q` is some spectral quantity (e.g. the BRF)
    * :math:`\lambda` denotes the wavelength
    * :math:`[\lambda_{\mathrm{min}},\, \lambda_{\mathrm{max}}]` is the
      wavelength bin

    Results
    -------

    Note: :math:`\mathrm{RAD}(x, y) = \left| \dfrac{x - y}{x} \right|`

    .. image:: generated/plots/test_onedim_ckd_vs_mono_1050_0.0.png
       :width: 95%
    .. image:: generated/plots/test_onedim_ckd_vs_mono_1050_0.5.png
       :width: 95%
    .. image:: generated/plots/test_onedim_ckd_vs_mono_1050_1.0.png
       :width: 95%
    """
    # Settings
    zeniths = np.linspace(-75, 75, 11)
    spp = 1e4
    wavelengths = np.linspace(1045.0, 1055.0, 1001) * ureg.nm  # mono mode setting
    bin_set = "10nm"  # ckd mode setting
    bins = "1050"  # ckd mode setting

    # Monochromatic experiment
    mono_exp = init_mono_experiment(
        wavelengths=wavelengths,
        spp=spp,
        reflectance=reflectance,
        zeniths=zeniths,
    )
    with uck.override(length="km"):  # this is a temporary workaround the 'batman' issue
        mono_exp.run()
    mono_results = mono_exp.results["measure"]

    wavelength_bin_width = (wavelengths[-1] - wavelengths[0]).m_as(mono_results.w.units)
    mono_results_averaged = mono_results.integrate(coord="w") / wavelength_bin_width

    # CKD experiment
    ckd_exp = init_ckd_experiment(
        bin_set=bin_set,
        bins=bins,
        spp=spp,
        reflectance=reflectance,
        zeniths=zeniths,
    )
    ckd_exp.run()
    ckd_results = ckd_exp.results["measure"]

    # Make figure
    filename = f"test_onedim_ckd_vs_mono_{bins}_{reflectance}.png"
    ensure_output_dir(os.path.join(output_dir, "plots"))
    fname_plot = os.path.join(output_dir, "plots", filename)
    make_figure(
        ckd_results=ckd_results,
        mono_results_averaged=mono_results_averaged,
        fname_plot=fname_plot,
        wavelength_bin="[1045, 1055] nm",
        reflectance=reflectance,
    )

    # Assert averaged mono results are consistent with ckd results
    mono_brf = mono_results_averaged.brf.squeeze().values
    ckd_brf = ckd_results.brf.squeeze().values
    assert np.allclose(mono_brf, ckd_brf, rtol=5e-2)