import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import eradiate
from eradiate.scenes.biosphere import DiscreteCanopy
from eradiate.units import unit_registry as ureg

matplotlib.rcParams["text.usetex"] = True

eradiate_dir = os.environ["ERADIATE_DIR"]
output_dir = os.path.join(eradiate_dir, "test_report", "generated")


def ensure_output_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


@pytest.mark.slow
@pytest.mark.parametrize("reflectance", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("sza", [0.0, 30.0, 60.0])
def test_compare_rami4atm_onedim(mode_mono_double, reflectance, sza):
    r"""
    Compare Rami4ATMExperiment with OneDimExperiment
    ================================================

    This test compares the results of a OneDimExperiment with a corresponding
    Rami4ATMExperiment run. To achieve that, the Rami4ATMExperiment is set up without an explicit
    canopy and a basic atmosphere.

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

    .. image:: generated/plots/test_compare_r4a_onedim_0.0_0.0.png
       :width: 30%
    .. image:: generated/plots/test_compare_r4a_onedim_0.0_0.5.png
       :width: 30%
    .. image:: generated/plots/test_compare_r4a_onedim_0.0_1.0.png
       :width: 30%

    .. image:: generated/plots/test_compare_r4a_onedim_30.0_0.0.png
       :width: 30%
    .. image:: generated/plots/test_compare_r4a_onedim_30.0_0.5.png
       :width: 30%
    .. image:: generated/plots/test_compare_r4a_onedim_30.0_1.0.png
       :width: 30%

    .. image:: generated/plots/test_compare_r4a_onedim_60.0_0.0.png
       :width: 30%
    .. image:: generated/plots/test_compare_r4a_onedim_60.0_0.5.png
       :width: 30%
    .. image:: generated/plots/test_compare_r4a_onedim_60.0_1.0.png
       :width: 30%
    """

    surface = eradiate.scenes.surface.surface_factory.convert(
        {"type": "lambertian", "reflectance": reflectance}
    )
    illumination = eradiate.scenes.illumination.illumination_factory.convert(
        {"type": "directional", "zenith": sza, "azimuth": 0.0}
    )
    atmosphere = eradiate.scenes.atmosphere.atmosphere_factory.convert(
        {
            "type": "heterogeneous",
            "molecular_atmosphere": {
                "type": "molecular",
                "has_absorption": False,
                "has_scattering": True,
            },
        }
    )
    measures = eradiate.scenes.measure.measure_factory.convert(
        {
            "type": "distant",
            "id": "measure",
            "construct": "from_viewing_angles",
            "zeniths": np.linspace(-90, 90, 19),
            "azimuths": 0.0,
            "spp": 1e4,
        }
    )

    onedim = eradiate.experiments.OneDimExperiment(
        surface=surface,
        illumination=illumination,
        atmosphere=atmosphere,
        measures=measures,
    )

    r4a = eradiate.experiments.Rami4ATMExperiment(
        surface=surface,
        illumination=illumination,
        atmosphere=atmosphere,
        measures=measures,
        canopy=None,
    )

    onedim.run()
    r4a.run()

    fig = plt.figure(figsize=(6, 3))
    ax = plt.gca()
    ax.plot(
        np.squeeze(onedim.results["measure"].vza.values),
        np.squeeze(onedim.results["measure"]["radiance"].values),
        label="onedim",
        marker=".",
        ls="--",
    )
    ax.plot(
        np.squeeze(r4a.results["measure"].vza.values),
        np.squeeze(r4a.results["measure"]["radiance"].values),
        label="r4a",
        marker=".",
        ls="--",
    )
    plt.xlabel("Signed viewing zenith angle [deg]")
    plt.xlim([-100, 150])
    plt.xticks([-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0])
    plt.ylabel("BRF [dimensionless]")
    plt.title(fr"sza = {sza} - $\rho$ = {reflectance}")
    plt.legend(loc="center right")

    filename = f"test_compare_r4a_onedim_{sza}_{reflectance}.png"
    ensure_output_dir(os.path.join(output_dir, "plots"))
    fname_plot = os.path.join(output_dir, "plots", filename)
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
def test_compare_rami4atm_rami(mode_mono_double, sza, lai):
    r"""
    Compare Rami4ATMExperiment with RamiExperiment
    ==============================================

    This test compares the results of a RamiExperiment with a corresponding
    Rami4ATMExperiment run. To achieve that, the Rami4ATMExperiment is set up without an atmosphere.

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

    .. image:: generated/plots/test_compare_r4a_rami_0.0_1.0.png
       :width: 30%
    .. image:: generated/plots/test_compare_r4a_rami_0.0_2.0.png
       :width: 30%
    .. image:: generated/plots/test_compare_r4a_rami_0.0_3.0.png
       :width: 30%

    .. image:: generated/plots/test_compare_r4a_rami_30.0_1.0.png
       :width: 30%
    .. image:: generated/plots/test_compare_r4a_rami_30.0_2.0.png
       :width: 30%
    .. image:: generated/plots/test_compare_r4a_rami_30.0_3.0.png
       :width: 30%

    .. image:: generated/plots/test_compare_r4a_rami_60.0_1.0.png
       :width: 30%
    .. image:: generated/plots/test_compare_r4a_rami_60.0_2.0.png
       :width: 30%
    .. image:: generated/plots/test_compare_r4a_rami_60.0_3.0.png
       :width: 30%
    """

    surface = eradiate.scenes.surface.surface_factory.convert(
        {"type": "lambertian", "reflectance": 0.25}
    )
    illumination = eradiate.scenes.illumination.illumination_factory.convert(
        {"type": "directional", "zenith": sza, "azimuth": 0.0}
    )
    canopy = DiscreteCanopy.homogeneous(
        lai=lai,
        leaf_radius=0.1 * ureg.m,
        l_horizontal=10.0 * ureg.m,
        l_vertical=2.0 * ureg.m,
        padding=5,
    )
    measures = eradiate.scenes.measure.measure_factory.convert(
        {
            "type": "distant",
            "id": "measure",
            "construct": "from_viewing_angles",
            "zeniths": np.linspace(-90, 90, 19),
            "azimuths": 0.0,
            "spp": 1e4,
        }
    )

    rami = eradiate.experiments.RamiExperiment(
        surface=surface,
        illumination=illumination,
        canopy=canopy,
        measures=measures,
    )

    r4a = eradiate.experiments.Rami4ATMExperiment(
        surface=surface,
        illumination=illumination,
        atmosphere=None,
        canopy=canopy,
        measures=measures,
    )

    rami.run()
    r4a.run()

    fig = plt.figure(figsize=(6, 3))
    ax = plt.gca()
    ax.plot(
        np.squeeze(rami.results["measure"].vza.values),
        np.squeeze(rami.results["measure"]["radiance"].values),
        label="rami",
        marker=".",
        ls="--",
    )
    ax.plot(
        np.squeeze(r4a.results["measure"].vza.values),
        np.squeeze(r4a.results["measure"]["radiance"].values),
        label="r4a",
        marker=".",
        ls="--",
    )
    plt.xlabel("Signed viewing zenith angle [deg]")
    plt.xlim([-100, 150])
    plt.xticks([-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0])
    plt.ylabel("BRF [dimensionless]")
    plt.title(f"sza = {sza} - LAI = {lai}")
    plt.legend(loc="center right")

    filename = f"test_compare_r4a_rami_{sza}_{lai}.png"
    ensure_output_dir(os.path.join(output_dir, "plots"))
    fname_plot = os.path.join(output_dir, "plots", filename)
    plt.tight_layout()
    fig.savefig(fname_plot, dpi=200)
    plt.close()

    assert np.all(
        rami.results["measure"]["radiance"].values
        == r4a.results["measure"]["radiance"].values
    )


matplotlib.rcParams["text.usetex"] = False
