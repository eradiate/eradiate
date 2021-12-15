"""Test cases with OneDimSolverApp and an RPV surface."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import eradiate
from eradiate import unit_registry as ureg


def map_to_0_360(x: float) -> float:
    return x % 360


@pytest.mark.parametrize("illumination_azimuth", [0.0, 30.0, 120.0, 210.0, 300.0])
def test_film_to_angular_coord_conversion_multi_distant(
    mode_mono_double, illumination_azimuth, artefact_dir
):
    r"""
    Film to angular coordinates conversion (`multi_distant`)
    ========================================================

    This test case asserts the correct conversion from film coordinates (x, y)
    to angular coordinates (theta, phi).

    We check that the conversion is correct by computing the BRF of a scene
    built such that it is forward/backward scattering and we verify that the
    BRF is consistent with this scattering behaviour.


    Rationale
    ---------

    * Geometry: a square surface with unit size and a RPV BRDF with
      reflectance :math:`\rho_0 = 0.1`.
    * Illumination: Directional illumination with a zenith angle
      :math:`\theta = 30.0°` and an azimuth angle
      :math:`\phi \in [0.0, 30.0, 120.0, 210.0, 300.0]°`.
    * Sensor: Distant reflectance measure covering a plane (20 angular points,
      1 sample per pixel).


    Expected behaviour
    ------------------

    Within the scope of this test, we define the forward region of the
    principal plane by:

    .. math:: \phi_v = \phi_i + 180°
       :label: multi_distant_forward

    where:

    * :math:`\phi_v` is the viewing azimuth angle [deg]
    * :math:`\phi_i` is the illumination azimuth angle [deg]

    Correspondingly, the backward region of the principal plane is defined by:

    .. math:: \phi_v = \phi_i
       :label: multi_distant_backward

    We expect that the mean BRF value will be larger/smaller in the forward
    region :eq:`multi_distant_forward` than in the backward region
    :eq:`multi_distant_backward` for a forward/backward scattering scene,
    respectively.

    Results
    -------

    .. image:: generated/plots/test_ftacc_multi_distant_forward_0.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_multi_distant_forward_30.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_multi_distant_forward_120.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_multi_distant_forward_210.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_multi_distant_forward_300.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_multi_distant_backward_0.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_multi_distant_backward_30.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_multi_distant_backward_120.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_multi_distant_backward_210.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_multi_distant_backward_300.0.png
       :width: 95%
    """

    def configure_experiment(g, illumination_azimuth):
        measure = {
            "id": "measure",
            "type": "distant",
            "construct": "from_viewing_angles",
            "azimuths": illumination_azimuth,
            "zeniths": np.linspace(-89, 89, n_vza) * ureg.deg,
            "spp": spp,
        }

        illumination = eradiate.scenes.illumination.DirectionalIllumination(
            zenith=30.0 * ureg.deg,
            azimuth=illumination_azimuth,
        )

        surface = eradiate.scenes.surface.RPVSurface(
            width=1.0,
            g=g,
            rho_0=reflectance,
            k=0.9,
        )

        integrator = eradiate.scenes.integrators.PathIntegrator()

        return eradiate.experiments.OneDimExperiment(
            measures=[measure],
            illumination=illumination,
            atmosphere=None,
            surface=surface,
            integrator=integrator,
        )

    spp = 1
    reflectance = 0.1
    n_vza = 20
    measure_azimuth = illumination_azimuth
    g_abs = 0.1
    g1 = g_abs
    g2 = -g_abs
    experiment1 = configure_experiment(
        g=g1,
        illumination_azimuth=illumination_azimuth,
    )
    experiment2 = configure_experiment(
        g=g2,
        illumination_azimuth=illumination_azimuth,
    )

    experiment1.run()
    experiment2.run()

    results1 = experiment1.results["measure"]
    results2 = experiment2.results["measure"]

    def select_forward_brf(brf, illumination_azimuth, measure_azimuth):
        relative_azimuth = map_to_0_360(measure_azimuth - illumination_azimuth)
        if relative_azimuth == 0.0:
            return brf.where(brf.vza < 0.0)
        elif relative_azimuth == 180.0:
            return brf.where(brf.vza > 0.0)
        else:
            raise ValueError(
                "cannot compute forward BRF when the relative azimuth is not "
                "either 0 or 180."
            )

    def select_backward_brf(brf, illumination_azimuth, measure_azimuth):
        relative_azimuth = map_to_0_360(measure_azimuth - illumination_azimuth)
        if relative_azimuth == 0.0:
            return brf.where(brf.vza > 0.0)
        elif relative_azimuth == 180.0:
            return brf.where(brf.vza < 0.0)
        else:
            raise ValueError(
                "cannot compute backward BRF when the relative azimuth is not "
                "either 0 or 180."
            )

    def is_forward_scattering(brf, illumination_azimuth, measure_azimuth):
        forward_brf = select_forward_brf(
            brf=brf,
            illumination_azimuth=illumination_azimuth,
            measure_azimuth=measure_azimuth,
        )
        backward_brf = select_backward_brf(
            brf=brf,
            illumination_azimuth=illumination_azimuth,
            measure_azimuth=measure_azimuth,
        )
        return forward_brf.mean().values > backward_brf.mean().values

    def make_figure(results, g, artefact_dir, forward=True):
        brf = results.brf
        brf_forward = select_forward_brf(
            brf=brf,
            illumination_azimuth=illumination_azimuth,
            measure_azimuth=measure_azimuth,
        )
        brf_backward = select_backward_brf(
            brf=brf,
            illumination_azimuth=illumination_azimuth,
            measure_azimuth=measure_azimuth,
        )

        brf_values = np.concatenate([brf_forward.values, brf_backward.values])
        brf_max = np.nanmax(brf_values)
        brf_min = np.nanmin(brf_values)
        ylim = [0.9 * brf_min, 1.1 * brf_max]

        if forward:
            desc = "Forward"
        else:
            desc = "Backward"

        fig = plt.figure(figsize=(8, 3))
        plt.suptitle(
            f"{desc} scattering RPV surface (g={g}), "
            f"illumination azimuth = {illumination_azimuth}°"
        )

        plt.subplot(1, 2, 1)
        ax = plt.gca()
        ax.set_xticks([-90, -60, -30, 0.0])
        ax.set_xticklabels(["-90", "-60", "-30", "0"])

        brf_forward.plot(
            x="vza",
            ls="dotted",
            marker=".",
            ylim=ylim,
            xlim=[-95, 0],
        )
        mean = float(brf_forward.mean().values)
        plt.text(
            s=f"mean = {mean:.2e}",
            x=-45,
            y=(brf_max + brf_min) / 2.0,
            ha="center",
            color="red",
        )
        plt.title("forward BRF")

        plt.subplot(1, 2, 2)
        ax = plt.gca()
        ax.set_xticks([0.0, 30.0, 60.0, 90.0])
        ax.set_xticklabels(["0", "30", "60", "90"])
        brf_backward.plot(
            x="vza",
            ls="dotted",
            marker=".",
            ylim=ylim,
            xlim=[0, 95],
        )
        mean = float(brf_backward.mean().values)
        plt.text(
            s=f"mean = {mean:.2e}",
            x=45,
            y=(brf_max + brf_min) / 2.0,
            ha="center",
            color="red",
        )
        plt.title("backward BRF")
        filename = f"test_ftacc_multi_distant_{desc.lower()}_{illumination_azimuth}.png"
        outdir = os.path.join(artefact_dir, "plots")
        os.makedirs(outdir, exist_ok=True)
        fname_plot = os.path.join(outdir, filename)
        plt.tight_layout()
        fig.savefig(fname_plot, dpi=200)
        plt.close()

    make_figure(results=results1, g=g1, artefact_dir=artefact_dir, forward=True)
    make_figure(results=results2, g=g2, artefact_dir=artefact_dir, forward=False)

    assert is_forward_scattering(
        brf=results1.brf,
        illumination_azimuth=illumination_azimuth,
        measure_azimuth=measure_azimuth,
    )
    assert not is_forward_scattering(
        brf=results2.brf,
        illumination_azimuth=illumination_azimuth,
        measure_azimuth=measure_azimuth,
    )


def where_azimuth(
    da: xr.DataArray, name: str, start: float, stop: float
) -> xr.DataArray:
    """
    Select data for viewing azumith in ['start', 'stop']°.
    """
    if start < stop:
        return da.where((da.vaa > start) & (da.vaa < stop))
    else:
        return xr.merge(
            [
                da.where((da.vaa > start) & (da.vaa < 360.0)),
                da.where((da.vaa > 0.0) & (da.vaa < stop)),
            ]
        )[name]


def select_forward(
    da: xr.DataArray, name: str, illumination_azimuth: float
) -> xr.DataArray:
    r"""
    Select hemispherical data in forward direction.

    If :math:`\phi_i` is the illumination azimuth and :math:`\phi_v` is the
    viewing azimuth, the forward region is defined by:

    .. math::

        \phi_i + 90° \ll \phi_v \ll \phi_i + 270°

    Parameters
    ----------
    da: DataArray
        Sector radiosity at an infinite distance in the hemisphere.

    name: str
        Name of the data variable to select.

    illumination_azimuth: float
        Azimuth [deg].

    Returns
    -------
    DataArray
        Data in forward direction.
    """
    return where_azimuth(
        da=da,
        name=name,
        start=map_to_0_360(illumination_azimuth + 90.0),
        stop=map_to_0_360(illumination_azimuth + 270.0),
    )


def select_backward(
    da: xr.DataArray, name: str, illumination_azimuth: float
) -> xr.DataArray:
    r"""
    Select hemispherical data in backward direction.

    If :math:`\phi_i` is the illumination azimuth and :math:`\phi_v` is the
    viewing azimuth, the backward region is defined by:

    .. math::

        \phi_i + 270° \ll \phi_v \ll \phi_i + 90°

    Parameters
    ----------
    da: DataArray
        Sector radiosity at an infinite distance in the hemisphere.

    name: str
        Name of the data variable to select.

    illumination_azimuth: float
        Azimuth [deg].

    Returns
    -------
    DataArray
        Backward sector radiosity.
    """
    return where_azimuth(
        da=da,
        name=name,
        start=map_to_0_360(illumination_azimuth + 270.0),
        stop=map_to_0_360(illumination_azimuth + 90.0),
    )


def is_forward_scattering(
    da: xr.DataArray, name: str, illumination_azimuth: float
) -> bool:
    """
    Determines if a scene is forward scattering.

    Parameters
    ----------
    da: DataArray
        Sector radiosity at an infinite distance in the hemisphere.

    name: str
        Name of the hemispherical variable.

    illumination_azimuth: float
        Azimuth [deg].
    """
    da_forward = select_forward(
        da=da, name=name, illumination_azimuth=illumination_azimuth
    )

    da_backward = select_backward(
        da=da, name=name, illumination_azimuth=illumination_azimuth
    )

    return da_forward.mean().values > da_backward.mean().values


def make_figure(
    results: xr.Dataset,
    name: str,
    g: float,
    forward: str,
    illumination_azimuth: float,
    measure: str,
    res: int,
    artefact_dir,
):
    if forward:
        desc = "Forward"
    else:
        desc = "Backward"

    fig = plt.figure(figsize=(8, 3))
    plt.suptitle(
        f"{desc} scattering RPV surface (g={g}), "
        f"illumination azimuth = {illumination_azimuth}°"
    )

    plt.subplot(1, 2, 1)
    ax = plt.gca()
    ax.set_aspect("equal")
    da_forward = select_forward(
        da=results[name],
        name=name,
        illumination_azimuth=illumination_azimuth,
    )
    da_forward.plot()
    mean = float(da_forward.mean().values)
    plt.text(s=f"mean = {mean:.2e}", x=res / 2, y=res / 2, ha="center", color="red")
    plt.title(f"forward {name}")

    plt.subplot(1, 2, 2)
    ax = plt.gca()
    ax.set_aspect("equal")
    da_backward = select_backward(
        da=results[name],
        name=name,
        illumination_azimuth=illumination_azimuth,
    )
    da_backward.plot()
    mean = float(da_backward.mean().values)
    plt.text(s=f"mean = {mean:.2e}", x=res / 2, y=res / 2, ha="center", color="red")
    plt.title(f"backward {name}")
    filename = f"test_ftacc_{measure}_{desc.lower()}_{illumination_azimuth}.png"
    outdir = os.path.join(artefact_dir, "plots")
    os.makedirs(outdir, exist_ok=True)
    fname_plot = os.path.join(outdir, filename)
    plt.tight_layout()
    fig.savefig(fname_plot, dpi=200)
    plt.close()


@pytest.mark.parametrize("illumination_azimuth", [0.0, 30.0, 120.0, 210.0, 300.0])
def test_film_to_angular_coord_conversion_distant_flux(
    mode_mono_double, illumination_azimuth, artefact_dir
):
    r"""
    Film to angular coordinates conversion (`distant_flux`)
    =======================================================

    This test case asserts the correct conversion from film coordinates (x, y)
    to angular coordinates (theta, phi) with the `distant_flux` measure.

    Since the mapping is only visible in post-processed results, this test
    requires the Experiment framework.

    We setup two experiments such that the first one includes a forward
    scattering scene and the second one a backward scattering scene.

    This is achieved using a surface reflectance model (`RPV`) that provides a
    parameter (`g`) to control the forward/backward scattering behaviour of the
    surface.

    We set a `distant_flux` measure to compute the sector radiosity at infinite
    distance in the hemisphere whose direction is the upward direction.

    Rationale
    ---------

    * Geometry: a square surface with unit size and a RPV BRDF with
      reflectance :math:`\rho_0 = 0.1` and with asymmetry parameter :math:`g`
      set to 0.1 and -0.1 for the forward and backward scattering scenes,
      respectively.
    * Illumination: Directional illumination with a zenith angle
      :math:`\theta_i = 30.0°` and an azimuth angle
      :math:`\phi_i \in [0.0, 30.0, 120.0, 210.0, 300.0]°`.
    * Sensor: Distant flux measure covering a hemisphere (32 * 32 angular
      points, 1 sample per pixel).


    Expected behaviour
    ------------------

    If :math:`\phi_i` is the illumination azimuth and :math:`\phi_v` is the
    viewing azimuth, we define the forward region of the hemisphere by:

    .. math:: \phi_i + 90° \lt \phi_v \lt \phi_i + 270°
       :label: distant_flux_forward

    and correspondingly, the backward region, by:

    .. math:: \phi_i + 270° \lt \phi_v \lt \phi_i + 90°
       :label: distant_flux_backward

    We then compute their mean value within the azimuth angle range where they
    are non zero.

    We expect that the forward scattering scene will yield to a measured sector
    radiosity being larger in the range of azimuth angles given by
    :eq:`distant_flux_forward` than in the range of azimuth angles given by
    :eq:`distant_flux_backward`.
    We expect the opposite result for the backward scattering scene.

    Results
    -------

    .. image:: generated/plots/test_ftacc_distant_flux_forward_0.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_distant_flux_forward_30.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_distant_flux_forward_120.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_distant_flux_forward_210.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_distant_flux_forward_300.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_distant_flux_backward_0.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_distant_flux_backward_30.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_distant_flux_backward_120.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_distant_flux_backward_210.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_distant_flux_backward_300.0.png
       :width: 95%
    """

    def configure_experiment(g, illumination_azimuth):
        measure = {
            "id": "measure",
            "type": "distant_flux",
            "direction": [0, 0, 1],
            "film_resolution": (res, res),
            "spp": spp,
        }

        illumination = eradiate.scenes.illumination.DirectionalIllumination(
            zenith=30.0 * ureg.deg,
            azimuth=illumination_azimuth,
        )

        surface = eradiate.scenes.surface.RPVSurface(
            width=1.0,
            g=g,
            rho_0=reflectance,
            k=0.9,
        )

        integrator = eradiate.scenes.integrators.PathIntegrator()

        return eradiate.experiments.OneDimExperiment(
            measures=[measure],
            illumination=illumination,
            atmosphere=None,
            surface=surface,
            integrator=integrator,
        )

    spp = 1
    res = 32
    reflectance = 0.1
    g_abs = 0.1
    g1 = g_abs
    g2 = -g_abs
    experiment1 = configure_experiment(
        g=g1,
        illumination_azimuth=illumination_azimuth,
    )
    experiment2 = configure_experiment(
        g=g2,
        illumination_azimuth=illumination_azimuth,
    )

    experiment1.run()
    experiment2.run()

    results1 = experiment1.results["measure"]
    results2 = experiment2.results["measure"]

    make_figure(
        results=results1,
        name="sector_radiosity",
        g=g1,
        forward=True,
        illumination_azimuth=illumination_azimuth,
        measure="distant_flux",
        res=res,
        artefact_dir=artefact_dir,
    )
    make_figure(
        results=results2,
        name="sector_radiosity",
        g=g2,
        forward=False,
        illumination_azimuth=illumination_azimuth,
        measure="distant_flux",
        res=res,
        artefact_dir=artefact_dir,
    )

    assert is_forward_scattering(
        da=results1.sector_radiosity,
        name="sector_radiosity",
        illumination_azimuth=illumination_azimuth,
    )
    assert not is_forward_scattering(
        da=results2.sector_radiosity,
        name="sector_radiosity",
        illumination_azimuth=illumination_azimuth,
    )


@pytest.mark.parametrize("illumination_azimuth", [0.0, 30.0, 120.0, 210.0, 300.0])
def test_film_to_angular_coord_conversion_hemispherical_distant(
    mode_mono_double, illumination_azimuth, artefact_dir
):
    r"""
    Film to angular coordinates conversion (`hdistant`)
    ===================================================

    This test case asserts the correct conversion from film coordinates (x, y)
    to angular coordinates (theta, phi) with the `hdistant` measure.

    Since the mapping is only visible in post-processed results, this test
    requires the Experiment framework.

    We setup two experiments such that the first one includes a forward
    scattering scene and the second one a backward scattering scene.

    This is achieved using a surface reflectance model (`RPV`) that provides a
    parameter (`g`) to control the forward/backward scattering behaviour of the
    surface.

    We set a `hemispherical_distant` measure to compute the BRF at infinite
    distance in the hemisphere whose direction is the upward direction.

    Rationale
    ---------

    * Geometry: a square surface with unit size and a RPV BRDF with
      reflectance :math:`\rho_0 = 0.1` and with asymmetry parameter :math:`g`
      set to 0.1 and -0.1 for the forward and backward scattering scenes,
      respectively.
    * Illumination: Directional illumination with a zenith angle
      :math:`\theta_i = 30.0°` and an azimuth angle
      :math:`\phi_i \in [0.0, 30.0, 120.0, 210.0, 300.0]°`.
    * Sensor: Distant hemispherical measure covering the hemisphere (32 * 32
      angular points, 1 sample per pixel).


    Expected behaviour
    ------------------

    If :math:`\phi_i` is the illumination azimuth and :math:`\phi_v` is the
    viewing azimuth, we define the forward region of the hemisphere by:

    .. math:: \phi_i + 90° \lt \phi_v \lt \phi_i + 270°
       :label: hemispherical_distant_forward

    and correspondingly, the backward region, by:

    .. math:: \phi_i + 270° \lt \phi_v \lt \phi_i + 90°
       :label: hemispherical_distant_backward

    We then compute their mean value within the azimuth angle range where they
    are non zero.

    We expect that the forward scattering scene will yield to a measured BRF
    being larger in the range of azimuth angles given by
    :eq:`hemispherical_distant_forward` than in the range of azimuth angles
    given by :eq:`hemispherical_distant_backward`.
    We expect the opposite result for the backward scattering scene.

    Results
    -------

    .. image:: generated/plots/test_ftacc_hdistant_forward_0.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_hdistant_forward_30.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_hdistant_forward_120.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_hdistant_forward_210.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_hdistant_forward_300.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_hdistant_backward_0.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_hdistant_backward_30.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_hdistant_backward_120.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_hdistant_backward_210.0.png
       :width: 95%
    .. image:: generated/plots/test_ftacc_hdistant_backward_300.0.png
       :width: 95%
    """

    def configure_experiment(g, illumination_azimuth):
        measure = {
            "id": "measure",
            "type": "hdistant",
            "direction": [0, 0, 1],
            "film_resolution": (res, res),
            "spp": spp,
        }

        illumination = eradiate.scenes.illumination.DirectionalIllumination(
            zenith=30.0 * ureg.deg,
            azimuth=illumination_azimuth,
        )

        surface = eradiate.scenes.surface.RPVSurface(
            width=1.0,
            g=g,
            rho_0=reflectance,
            k=0.9,
        )

        integrator = eradiate.scenes.integrators.PathIntegrator()

        return eradiate.experiments.OneDimExperiment(
            measures=[measure],
            illumination=illumination,
            atmosphere=None,
            surface=surface,
            integrator=integrator,
        )

    spp = 1
    res = 32
    reflectance = 0.1
    g_abs = 0.1
    g1 = g_abs
    g2 = -g_abs
    experiment1 = configure_experiment(
        g=g1,
        illumination_azimuth=illumination_azimuth,
    )
    experiment2 = configure_experiment(
        g=g2,
        illumination_azimuth=illumination_azimuth,
    )

    experiment1.run()
    experiment2.run()

    results1 = experiment1.results["measure"]
    results2 = experiment2.results["measure"]

    make_figure(
        results=results1,
        name="brf",
        g=g1,
        forward=True,
        illumination_azimuth=illumination_azimuth,
        measure="hdistant",
        res=res,
        artefact_dir=artefact_dir,
    )
    make_figure(
        results=results2,
        name="brf",
        g=g2,
        forward=False,
        illumination_azimuth=illumination_azimuth,
        measure="hdistant",
        res=res,
        artefact_dir=artefact_dir,
    )

    assert is_forward_scattering(
        da=results1.brf,
        name="brf",
        illumination_azimuth=illumination_azimuth,
    )
    assert not is_forward_scattering(
        da=results2.brf,
        name="brf",
        illumination_azimuth=illumination_azimuth,
    )
