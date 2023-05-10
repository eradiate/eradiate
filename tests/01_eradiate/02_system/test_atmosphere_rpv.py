"""Test cases with AtmosphereExperiment and an RPV surface."""

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
    mode_mono, illumination_azimuth, artefact_dir
):
    r"""
    Film to angular coordinates conversion (``multi_distant``)
    ==========================================================

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
      :math:`\varphi \in [0.0, 30.0, 120.0, 210.0, 300.0]°`.
    * Sensor: Distant reflectance measure covering a plane (20 angular points,
      1 sample per pixel).


    Expected behaviour
    ------------------

    Within the scope of this test, we define the forward region of the
    principal plane by:

    .. math:: \varphi_v = \varphi_i + 180°
       :label: multi_distant_forward

    where:

    * :math:`\varphi_v` is the viewing azimuth angle [deg]
    * :math:`\varphi_i` is the illumination azimuth angle [deg]

    Correspondingly, the backward region of the principal plane is defined by:

    .. math:: \varphi_v = \varphi_i
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

    spp = 1
    reflectance = 0.1
    n_vza = 20
    measure_azimuth = illumination_azimuth
    gs = {"forward": 0.1, "backward": -0.1}

    experiments = {
        orientation: eradiate.experiments.AtmosphereExperiment(
            measures={
                "type": "mdistant",
                "construct": "hplane",
                "azimuth": illumination_azimuth,
                "zeniths": np.linspace(-89, 89, n_vza) * ureg.deg,
                "spp": spp,
            },
            illumination={
                "type": "directional",
                "zenith": 30.0 * ureg.deg,
                "azimuth": illumination_azimuth,
            },
            atmosphere=None,
            surface={
                "type": "rpv",
                "g": g,
                "rho_0": reflectance,
                "k": 0.9,
            },
        )
        for orientation, g in gs.items()
    }

    results = {name: eradiate.run(exp) for name, exp in experiments.items()}

    def select_brf(brf, orientation):
        relative_azimuth = map_to_0_360(measure_azimuth - illumination_azimuth)

        if relative_azimuth == 0.0:
            return (
                brf.where(brf.vza < 0.0)
                if orientation == "forward"
                else brf.where(brf.vza > 0.0)
            )
        elif relative_azimuth == 180.0:
            return (
                brf.where(brf.vza > 0.0)
                if orientation == "forward"
                else brf.where(brf.vza < 0.0)
            )
        else:
            raise ValueError(
                "cannot compute forward BRF when the relative azimuth is not "
                "either 0 or 180."
            )

    def is_forward_scattering(brf):
        forward_brf = select_brf(brf, "forward")
        backward_brf = select_brf(brf, "backward")
        return forward_brf.mean().values > backward_brf.mean().values

    def make_figure(brf, g, artefact_dir, orientation: str):
        brf_forward = select_brf(brf, "forward")
        brf_backward = select_brf(brf, "backward")

        brf_values = np.concatenate([brf_forward.values, brf_backward.values])
        brf_max = np.nanmax(brf_values)
        brf_min = np.nanmin(brf_values)
        ylim = [0.9 * brf_min, 1.1 * brf_max]

        desc = orientation.title()

        fig = plt.figure(figsize=(8, 3))
        plt.suptitle(
            f"{desc} scattering RPV surface (g={g}), "
            f"illumination azimuth = {illumination_azimuth}°"
        )

        with plt.rc_context({"lines.linestyle": ":", "lines.marker": "."}):
            for orientation, brf, sub, ticks, xtext in zip(
                ["forward", "backward"],
                [brf_forward, brf_backward],
                [(1, 2, 1), (1, 2, 2)],
                [[-90, -60, -30, 0], [0, 30, 60, 90]],
                [-45, 45],
            ):
                plt.subplot(*sub)
                ax = plt.gca()
                ax.set_xticks(ticks)
                ax.set_xticklabels(list(map(str, ticks)))
                brf.plot(x="vza", xlim=[-95, 0], ylim=ylim)
                mean = float(brf.mean().values)
                plt.text(
                    s=f"mean = {mean:.2e}",
                    x=xtext,
                    y=(brf_max + brf_min) / 2.0,
                    ha="center",
                    color="red",
                )
                plt.title(f"{orientation} BRF")

        filename = f"test_ftacc_multi_distant_{desc.lower()}_{illumination_azimuth}.png"
        outdir = os.path.join(artefact_dir, "plots")
        os.makedirs(outdir, exist_ok=True)
        fname_plot = os.path.join(outdir, filename)
        plt.tight_layout()
        fig.savefig(fname_plot, dpi=200)
        plt.close()

    for orientation, result in results.items():
        make_figure(result.brf, gs[orientation], artefact_dir, orientation)

    assert is_forward_scattering(results["forward"].brf)
    assert not is_forward_scattering(brf=results["backward"].brf)


def where_azimuth(
    da: xr.DataArray, name: str, start: float, stop: float
) -> xr.DataArray:
    """
    Select data for viewing azimuth in ['start', 'stop']°.
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


def select_orientation(
    da: xr.DataArray,
    name: str,
    illumination_azimuth: float,
    orientation: str,
) -> xr.DataArray:
    r"""
    Select hemispherical data in requested orientation.

    If :math:`\varphi_i` is the illumination azimuth and :math:`\varphi_v` is the
    viewing azimuth, the forward region is defined by:

    .. math::

        \varphi_i + 90° \ll \varphi_v \ll \varphi_i + 270°

    Parameters
    ----------
    da: DataArray
        Sector radiosity at an infinite distance in the hemisphere.

    name: str
        Name of the data variable to select.

    illumination_azimuth: float
        Azimuth [deg].

    orientation : {"forward", "backward"}
        Selected orientation.

    Returns
    -------
    DataArray
        Data in forward direction.
    """
    return (
        where_azimuth(
            da=da,
            name=name,
            start=map_to_0_360(illumination_azimuth + 90.0),
            stop=map_to_0_360(illumination_azimuth + 270.0),
        )
        if orientation == "forward"
        else where_azimuth(
            da=da,
            name=name,
            start=map_to_0_360(illumination_azimuth + 270.0),
            stop=map_to_0_360(illumination_azimuth + 90.0),
        )
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
    da_forward = select_orientation(da, name, illumination_azimuth, "forward")
    da_backward = select_orientation(da, name, illumination_azimuth, "backward")
    return da_forward.mean().values > da_backward.mean().values


def make_figure(
    results: xr.Dataset,
    name: str,
    g: float,
    forward: bool,
    illumination_azimuth: float,
    measure: str,
    res: int,
    artefact_dir,
):
    fig = plt.figure(figsize=(8, 3))

    desc = "Forward" if forward else "Backward"
    plt.suptitle(
        f"{desc} scattering RPV surface (g={g}), "
        f"illumination azimuth = {illumination_azimuth}°"
    )

    for orientation, sub in zip(
        ["forward", "backward"],
        [(1, 2, 1), (1, 2, 2)],
    ):
        plt.subplot(*sub)
        ax = plt.gca()
        ax.set_aspect("equal")
        da = select_orientation(
            results[name],
            name,
            illumination_azimuth,
            "forward",
        )
        da.plot()

        mean = float(da.mean().values)
        plt.text(
            s=f"mean = {mean:.2e}",
            x=res / 2,
            y=res / 2,
            ha="center",
            color="red",
        )
        plt.title(f"{orientation} {name}")

    filename = f"test_ftacc_{measure}_{desc.lower()}_{illumination_azimuth}.png"
    outdir = os.path.join(artefact_dir, "plots")
    os.makedirs(outdir, exist_ok=True)
    fname_plot = os.path.join(outdir, filename)
    plt.tight_layout()
    fig.savefig(fname_plot, dpi=200)
    plt.close()


@pytest.mark.parametrize("illumination_azimuth", [0.0, 30.0, 120.0, 210.0, 300.0])
def test_film_to_angular_coord_conversion_distant_flux(
    mode_mono, illumination_azimuth, artefact_dir
):
    r"""
    Film to angular coordinates conversion (``distant_flux``)
    =========================================================

    This test case asserts the correct conversion from film coordinates (x, y)
    to angular coordinates (theta, phi) with the ``distant_flux`` measure.

    Since the mapping is only visible in post-processed results, this test
    requires the Experiment framework.

    We setup two experiments such that the first one includes a forward
    scattering scene and the second one a backward scattering scene.

    This is achieved using a surface reflectance model (``RPV``) that provides a
    parameter (``g``) to control the forward/backward scattering behaviour of the
    surface.

    We set a ``distant_flux`` measure to compute the sector radiosity at infinite
    distance in the hemisphere whose direction is the upward direction.

    Rationale
    ---------

    * Geometry: a square surface with unit size and a RPV BRDF with
      reflectance :math:`\rho_0 = 0.1` and with asymmetry parameter :math:`g`
      set to 0.1 and -0.1 for the forward and backward scattering scenes,
      respectively.
    * Illumination: Directional illumination with a zenith angle
      :math:`\theta_i = 30.0°` and an azimuth angle
      :math:`\varphi_i \in [0.0, 30.0, 120.0, 210.0, 300.0]°`.
    * Sensor: Distant flux measure covering a hemisphere (32 * 32 angular
      points, 1 sample per pixel).


    Expected behaviour
    ------------------

    If :math:`\varphi_i` is the illumination azimuth and :math:`\varphi_v` is the
    viewing azimuth, we define the forward region of the hemisphere by:

    .. math:: \phi_i + 90° < \phi_v < \phi_i + 270°
       :label: distant_flux_forward

    and correspondingly, the backward region, by:

    .. math:: \phi_i + 270° < \phi_v < \phi_i + 90°
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

    spp = 1
    res = 32
    reflectance = 0.1
    gs = {"forward": 0.1, "backward": -0.1}

    experiments = {
        orientation: eradiate.experiments.AtmosphereExperiment(
            measures={
                "type": "distant_flux",
                "direction": [0, 0, 1],
                "film_resolution": (res, res),
                "spp": spp,
            },
            illumination={
                "type": "directional",
                "zenith": 30.0 * ureg.deg,
                "azimuth": illumination_azimuth,
            },
            atmosphere=None,
            surface={"type": "rpv", "g": g, "rho_0": reflectance, "k": 0.9},
        )
        for orientation, g in gs.items()
    }

    results = {
        orientation: eradiate.run(exp) for orientation, exp in experiments.items()
    }

    for orientation, g in gs.items():
        make_figure(
            results=results[orientation],
            name="sector_radiosity",
            g=g,
            forward=(orientation == "forward"),
            illumination_azimuth=illumination_azimuth,
            measure="distant_flux",
            res=res,
            artefact_dir=artefact_dir,
        )

    assert is_forward_scattering(
        da=results["forward"].sector_radiosity,
        name="sector_radiosity",
        illumination_azimuth=illumination_azimuth,
    )
    assert not is_forward_scattering(
        da=results["backward"].sector_radiosity,
        name="sector_radiosity",
        illumination_azimuth=illumination_azimuth,
    )


@pytest.mark.parametrize("illumination_azimuth", [0.0, 30.0, 120.0, 210.0, 300.0])
def test_film_to_angular_coord_conversion_hemispherical_distant(
    mode_mono_double, illumination_azimuth, artefact_dir
):
    r"""
    Film to angular coordinates conversion (``hdistant``)
    =====================================================

    This test case asserts the correct conversion from film coordinates (x, y)
    to angular coordinates (theta, phi) with the ``hdistant`` measure.

    Since the mapping is only visible in post-processed results, this test
    requires the Experiment framework.

    We setup two experiments such that the first one includes a forward
    scattering scene and the second one a backward scattering scene.

    This is achieved using a surface reflectance model (``RPV``) that provides a
    parameter (``g``) to control the forward/backward scattering behaviour of the
    surface.

    We set a ``hemispherical_distant`` measure to compute the BRF at infinite
    distance in the hemisphere whose direction is the upward direction.

    Rationale
    ---------

    * Geometry: a square surface with unit size and a RPV BRDF with
      reflectance :math:`\rho_0 = 0.1` and with asymmetry parameter :math:`g`
      set to 0.1 and -0.1 for the forward and backward scattering scenes,
      respectively.
    * Illumination: Directional illumination with a zenith angle
      :math:`\theta_i = 30.0°` and an azimuth angle
      :math:`\varphi_i \in [0.0, 30.0, 120.0, 210.0, 300.0]°`.
    * Sensor: Distant hemispherical measure covering the hemisphere (32 * 32
      angular points, 1 sample per pixel).


    Expected behaviour
    ------------------

    If :math:`\varphi_i` is the illumination azimuth and :math:`\varphi_v` is the
    viewing azimuth, we define the forward region of the hemisphere by:

    .. math:: \phi_i + 90° < \phi_v < \phi_i + 270°
       :label: hemispherical_distant_forward

    and correspondingly, the backward region, by:

    .. math:: \phi_i + 270° < \phi_v < \phi_i + 90°
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

    spp = 1
    res = 32
    reflectance = 0.1
    gs = {"forward": 0.1, "backward": -0.1}

    experiments = {
        orientation: eradiate.experiments.AtmosphereExperiment(
            measures={
                "type": "hdistant",
                "direction": [0, 0, 1],
                "film_resolution": (res, res),
                "spp": spp,
            },
            illumination={
                "type": "directional",
                "zenith": 30.0 * ureg.deg,
                "azimuth": illumination_azimuth,
            },
            atmosphere=None,
            surface={"type": "rpv", "g": g, "rho_0": reflectance, "k": 0.9},
        )
        for orientation, g in gs.items()
    }

    results = {
        orientation: eradiate.run(exp) for orientation, exp in experiments.items()
    }

    for orientation, g in gs.items():
        make_figure(
            results=results[orientation],
            name="brf",
            g=g,
            forward=(orientation == "forward"),
            illumination_azimuth=illumination_azimuth,
            measure="hdistant",
            res=res,
            artefact_dir=artefact_dir,
        )

    assert is_forward_scattering(
        da=results["forward"].brf,
        name="brf",
        illumination_azimuth=illumination_azimuth,
    )
    assert not is_forward_scattering(
        da=results["backward"].brf,
        name="brf",
        illumination_azimuth=illumination_azimuth,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "atmosphere", [None, "homogeneous"], ids=["none", "homogeneous"]
)
@pytest.mark.parametrize("reflectance", [0.0, 0.5, 1.0])
def test_rpv_vs_lambertian(mode_mono, atmosphere, reflectance, artefact_dir, request):
    r"""
    RPV(:math:`\rho, g=0, k=1, rho_c=1`) equivalent to Lambertian(:math:`\rho`)
    ===========================================================================

    A surface with a lambertian BSDF behaves like the same surface but with a
    RPV BSDF with identical reflectance value, :math:`g = 0`, :math:`k = 1`
    and :math:`\rho_c = 1`.

    Rationale
    ---------

    * Geometry: a square surface with unit size and

      * experiment 1: a lambertian BSDF with reflectance :math:`\rho_0 = 1.0`.
      * experiment 2: a RPV BSDF with :math:`\rho_0 = 1.0`, :math:`g = 0`,
        :math:`k = 1`, :math:`rho_c = 1`.

    * Atmosphere: No atmosphere or default homogeneous atmosphere.
    * Illumination: Directional illumination with a zenith angle
      :math:`\theta = 30.0°` and an azimuth angle :math:`\varphi = 0.0`.
    * Sensor: Distant reflectance measure covering a plane (11 angular points,
      1 sample per pixel with no atmosphere, 10000 with atmosphere).

    Expected behaviour
    ------------------

    Experiment 1 and 2 TOA BRF values are equal.

    Results
    -------

    .. image:: generated/plots/test_rpv_vs_lambertian-none-0.0.png
       :width: 95%
    .. image:: generated/plots/test_rpv_vs_lambertian-none-0.5.png
       :width: 95%
    .. image:: generated/plots/test_rpv_vs_lambertian-none-1.0.png
       :width: 95%

    .. image:: generated/plots/test_rpv_vs_lambertian-homogeneous-0.0.png
       :width: 95%
    .. image:: generated/plots/test_rpv_vs_lambertian-homogeneous-0.5.png
       :width: 95%
    .. image:: generated/plots/test_rpv_vs_lambertian-homogeneous-1.0.png
       :width: 95%
    """

    def make_figure_rpv_vs_lambertian(fname_plot, results: dict, title=""):
        fig = plt.figure(figsize=(8, 3))
        results_lambertian = results["lambertian"]
        results_rpv = results["rpv"]

        with plt.rc_context({"lines.linestyle": "dashed"}):
            results_lambertian.brf.plot(x="vza", marker="o", label="lambertian")
            results_rpv.brf.plot(x="vza", marker="^", label="rpv")

        bias = np.zeros_like(results_lambertian.brf.values)
        np.divide(
            results_rpv.brf.values - results_lambertian.brf.values,
            results_lambertian.brf.values,
            where=results_lambertian.brf.values != 0,
            out=bias,
        )
        mean_bias = np.mean(np.abs(bias))
        plt.annotate(
            text=f"Mean bias: {round(100 * mean_bias, 2)} %",
            xy=(0.5, 0.5),
            horizontalalignment="center",
            xycoords="axes fraction",
        )
        plt.title(title)
        plt.tight_layout()

        print(f"Saving figure to {fname_plot}")
        fig.savefig(fname_plot, dpi=200)
        plt.close()

    bsdfs = {
        "lambertian": {"type": "lambertian", "reflectance": reflectance},
        "rpv": {
            "type": "rpv",
            **dict(
                rho_0=reflectance,
                g=0.0,
                k=1.0,
                rho_c=1.0,
            ),
        },
    }
    experiments = {
        bsdf: eradiate.experiments.AtmosphereExperiment(
            illumination={"type": "directional", "zenith": 30.0 * ureg.deg},
            measures={
                "type": "mdistant",
                "construct": "hplane",
                "zeniths": np.arange(-75, 75, 11),
                "azimuth": 0.0 * ureg.deg,
                "spp": 1 if atmosphere is None else 100000,
            },
            atmosphere=None if atmosphere is None else {"type": atmosphere},
            surface=bsdfs[bsdf],
        )
        for bsdf in bsdfs.keys()
    }

    # Run experiments
    results = {bsdf: eradiate.run(exp) for bsdf, exp in experiments.items()}

    # Make figure
    filename = f"{request.node.originalname}-{'none' if atmosphere is None else 'homogeneous'}-{reflectance}.png"
    outdir = os.path.join(artefact_dir, "plots")
    os.makedirs(outdir, exist_ok=True)
    fname_plot = os.path.join(outdir, filename)
    title = (
        f"{reflectance=}, no atmosphere"
        if atmosphere is None
        else f"{reflectance=}, homogeneous atmosphere"
    )
    make_figure_rpv_vs_lambertian(fname_plot, results, title=title)

    # assert result BRF values are equal
    lambertian = np.squeeze(results["lambertian"].brf.values)
    rpv = np.squeeze(results["rpv"].brf.values)

    if atmosphere is None:
        np.testing.assert_array_equal(rpv, lambertian)
    else:
        np.testing.assert_allclose(rpv, lambertian, rtol=1e-2, atol=2e-3)
