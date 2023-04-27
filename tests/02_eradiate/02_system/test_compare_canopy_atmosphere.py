import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eradiate
import eradiate.experiments as ertxp
import eradiate.scenes as ertsc
from eradiate.units import symbol
from eradiate.units import unit_registry as ureg


@pytest.mark.slow
@pytest.mark.parametrize("reflectance", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("sza", [0.0, 30.0, 60.0])
def test_compare_canopy_atmosphere_vs_atmosphere(
    mode_ckd_double, reflectance, sza, artefact_dir, ert_seed_state
):
    r"""
    Compare CanopyAtmosphereExperiment with AtmosphereExperiment
    ============================================================

    This test compares the results of an AtmosphereExperiment with a
    corresponding CanopyAtmosphereExperiment run. To achieve that, the
    CanopyAtmosphereExperiment is set up without an explicit canopy and a basic
    atmosphere.

    Rationale
    ---------

    Both experiments are set up with the same surface and atmosphere:

    * Geometry: a square surface with unit size and a Lambertian BRDF with
      reflectance :math:`\rho \in [0.0, 0.5, 1.0]`.
    * Illumination: Directional illumination with a zenith angle
      :math:`\theta \in [0.0, 30.0, 60.0]°`.
    * Atmosphere: Molecular atmosphere using the ussa1976 profile
    * Sensor: Distant reflectance measure covering a plane (21 angular points,
      10000 sample per pixel)
      We take an odd number of angular points to ensure that the special value
      :math:`\theta = 0°` is included.

    Expected behaviour
    ------------------

    The results of the two runs must be exactly equal.

    Results
    -------

    .. image:: generated/plots/test_compare_canopy_atmosphere_vs_atmosphere_0.0_0.0.png
       :width: 30%
    .. image:: generated/plots/test_compare_canopy_atmosphere_vs_atmosphere_0.0_0.5.png
       :width: 30%
    .. image:: generated/plots/test_compare_canopy_atmosphere_vs_atmosphere_0.0_1.0.png
       :width: 30%

    .. image:: generated/plots/test_compare_canopy_atmosphere_vs_atmosphere_30.0_0.0.png
       :width: 30%
    .. image:: generated/plots/test_compare_canopy_atmosphere_vs_atmosphere_30.0_0.5.png
       :width: 30%
    .. image:: generated/plots/test_compare_canopy_atmosphere_vs_atmosphere_30.0_1.0.png
       :width: 30%

    .. image:: generated/plots/test_compare_canopy_atmosphere_vs_atmosphere_60.0_0.0.png
       :width: 30%
    .. image:: generated/plots/test_compare_canopy_atmosphere_vs_atmosphere_60.0_0.5.png
       :width: 30%
    .. image:: generated/plots/test_compare_canopy_atmosphere_vs_atmosphere_60.0_1.0.png
       :width: 30%
    """

    surface = {"type": "lambertian", "reflectance": reflectance}
    illumination = {"type": "directional", "zenith": sza, "azimuth": 0.0}
    atmosphere = {
        "type": "heterogeneous",
        "molecular_atmosphere": {
            "type": "molecular",
            "has_absorption": False,
            "has_scattering": True,
        },
    }
    measure = {
        "type": "distant",
        "id": "measure",
        "construct": "hplane",
        "zeniths": np.arange(-90, 91, 15),
        "azimuth": 0.0,
        "spp": 1e4,
        "srf": ertsc.spectra.MultiDeltaSpectrum(
            wavelengths=550 * ureg.nm,
        ),
    }

    onedim = ertxp.AtmosphereExperiment(
        surface=surface,
        illumination=illumination,
        atmosphere=atmosphere,
        measures=measure,
    )

    r4a = ertxp.CanopyAtmosphereExperiment(
        surface=surface,
        illumination=illumination,
        atmosphere=atmosphere,
        measures=measure,
        canopy=None,
    )

    ert_seed_state.reset()
    eradiate.run(onedim, seed_state=ert_seed_state)

    ert_seed_state.reset()
    eradiate.run(r4a, seed_state=ert_seed_state)

    fig = plt.figure(figsize=(6, 3))
    ax = plt.gca()

    onedim_vza = np.squeeze(onedim.results["measure"].vza.values)
    onedim_radiance = np.squeeze(onedim.results["measure"]["radiance"].values)
    ax.plot(onedim_vza, onedim_radiance, label="onedim", marker=".", ls="--")

    r4a_vza = np.squeeze(r4a.results["measure"].vza.values)
    r4a_radiance = np.squeeze(r4a.results["measure"]["radiance"].values)
    ax.plot(r4a_vza, r4a_radiance, label="r4a", marker=".", ls="--")

    radiance_units = symbol(r4a.results["measure"]["radiance"].attrs["units"])
    plt.xlabel("Signed viewing zenith angle [°]")
    plt.xticks([-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0])
    plt.ylabel(f"Radiance [{radiance_units}]")
    plt.title(rf"SZA = {sza} — $\rho$ = {reflectance}")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    filename = f"test_compare_canopy_atmosphere_vs_atmosphere_{sza}_{reflectance}.png"
    outdir = os.path.join(artefact_dir, "plots")
    os.makedirs(outdir, exist_ok=True)
    fname_plot = os.path.join(outdir, filename)
    plt.tight_layout()
    fig.savefig(fname_plot, dpi=200)
    plt.close()

    assert np.all(
        onedim.results["measure"]["radiance"].values
        == r4a.results["measure"]["radiance"].values
    )


@pytest.mark.slow
@pytest.mark.parametrize("sza", [0.0, 30.0, 60.0])
@pytest.mark.parametrize("lai", [1.0, 2.0, 3.0])
def test_compare_canopy_atmosphere_vs_canopy(
    mode_mono_double, sza, lai, artefact_dir, ert_seed_state
):
    r"""
    Compare CanopyAtmosphereExperiment with CanopyExperiment
    ========================================================

    This test compares the results of a CanopyExperiment with a corresponding
    CanopyAtmosphereExperiment run. To achieve that, the
    CanopyAtmosphereExperiment is set up without an atmosphere.

    Rationale
    ---------

    Both experiments are set up with the same surface and atmosphere:

    * Geometry: a square surface with unit size and a Lambertian BRDF with
      reflectance :math:`\rho = 0.25`.
    * Illumination: Directional illumination with a zenith angle
      :math:`\theta \in [0.0, 30.0, 60.0]°`.
    * Atmosphere: None
    * Canopy: Homogeneous discrete canopy with LAI :math:`[1.0, 2.0, 3.0]`
    * Sensor: Distant reflectance measure covering a plane (21 angular points,
      10000 sample per pixel)
      We take an odd number of angular points to ensure that the special value
      :math:`\theta = 0°` is included.

    Expected behaviour
    ------------------

    The results of the two runs must be exactly equal.

    Results
    -------

    .. image:: generated/plots/test_compare_canopy_atmosphere_vs_canopy_0.0_1.0.png
       :width: 30%
    .. image:: generated/plots/test_compare_canopy_atmosphere_vs_canopy_0.0_2.0.png
       :width: 30%
    .. image:: generated/plots/test_compare_canopy_atmosphere_vs_canopy_0.0_3.0.png
       :width: 30%

    .. image:: generated/plots/test_compare_canopy_atmosphere_vs_canopy_30.0_1.0.png
       :width: 30%
    .. image:: generated/plots/test_compare_canopy_atmosphere_vs_canopy_30.0_2.0.png
       :width: 30%
    .. image:: generated/plots/test_compare_canopy_atmosphere_vs_canopy_30.0_3.0.png
       :width: 30%

    .. image:: generated/plots/test_compare_canopy_atmosphere_vs_canopy_60.0_1.0.png
       :width: 30%
    .. image:: generated/plots/test_compare_canopy_atmosphere_vs_canopy_60.0_2.0.png
       :width: 30%
    .. image:: generated/plots/test_compare_canopy_atmosphere_vs_canopy_60.0_3.0.png
       :width: 30%
    """

    surface = {"type": "lambertian", "reflectance": 0.25}
    illumination = {"type": "directional", "zenith": sza, "azimuth": 0.0}
    canopy = ertsc.biosphere.DiscreteCanopy.homogeneous(
        lai=lai,
        leaf_radius=0.1 * ureg.m,
        l_horizontal=10.0 * ureg.m,
        l_vertical=2.0 * ureg.m,
        padding=5,
    )
    measure = {
        "type": "distant",
        "id": "measure",
        "construct": "hplane",
        "zeniths": np.arange(-90, 91, 15),
        "azimuth": 0.0,
        "spp": 1e4,
        "srf": ertsc.spectra.MultiDeltaSpectrum(wavelengths=550.0 * ureg.nm),
    }

    rami = ertxp.CanopyExperiment(
        surface=surface,
        illumination=illumination,
        canopy=canopy,
        measures=measure,
    )

    r4a = ertxp.CanopyAtmosphereExperiment(
        surface=surface,
        illumination=illumination,
        atmosphere=None,
        canopy=canopy,
        measures=measure,
    )

    ert_seed_state.reset()
    eradiate.run(rami, seed_state=ert_seed_state)

    ert_seed_state.reset()
    eradiate.run(r4a, seed_state=ert_seed_state)

    fig = plt.figure(figsize=(6, 3))
    ax = plt.gca()

    rami_vza = np.squeeze(rami.results["measure"].vza.values)
    rami_radiance = np.squeeze(rami.results["measure"]["radiance"].values)
    ax.plot(rami_vza, rami_radiance, label="rami", marker=".", ls="--")

    r4a_vza = np.squeeze(r4a.results["measure"].vza.values)
    r4a_radiance = np.squeeze(r4a.results["measure"]["radiance"].values)
    ax.plot(r4a_vza, r4a_radiance, label="r4a", marker=".", ls="--")

    radiance_units = symbol(r4a.results["measure"]["radiance"].attrs["units"])
    plt.xlabel("Signed viewing zenith angle [°]")
    plt.xticks([-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0])
    plt.ylabel(f"Radiance [{radiance_units}]")
    plt.title(f"SZA = {sza} — LAI = {lai}")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    filename = f"test_compare_canopy_atmosphere_vs_canopy_{sza}_{lai}.png"
    outdir = os.path.join(artefact_dir, "plots")
    os.makedirs(outdir, exist_ok=True)
    fname_plot = os.path.join(outdir, filename)
    plt.tight_layout()
    fig.savefig(fname_plot, dpi=200)
    plt.close()

    assert np.all(
        rami.results["measure"]["radiance"].values
        == r4a.results["measure"]["radiance"].values
    )
