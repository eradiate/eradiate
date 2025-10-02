import attr
import matplotlib.pyplot as plt
import numpy as np
import pytest
from robot.api import logger

import eradiate
from eradiate import fresolver
from eradiate.test_tools.regression import RMSETest, SidakTTest, figure_to_html
from eradiate.test_tools.test_cases import rami4atm

brf_cases = [c for c in rami4atm.registry if c != "hom00_bla_a00s_m04_z30a000_brfpp"]
boa_cases = ["hom00_whi_s00s_m04_z30a000_brfpp", "hom00_rpv_e00s_m04_z30a000_brfpp"]
boa_thresholds = [5e-3, 5e-5]


@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.parametrize("case", brf_cases)
def test_rami4atm_brf(mode_ckd_double, case, artefact_dir):
    ctor = rami4atm.registry[case]
    exp = ctor(spp=1000)
    result = eradiate.run(exp)
    logger.info(result._repr_html_(), html=True)

    reference = fresolver.load_dataset(
        f"tests/regression_test_references/rami4atm/{case}-ref.nc"
    )

    test = SidakTTest(
        name=case,
        value=result,
        reference=reference,
        threshold=0.01,
        archive_dir=artefact_dir,
        variable="radiance",
        plot=False,
    )

    passed = test.run(diagnostic=True)

    assert passed


@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.parametrize("case,threshold", list(zip(boa_cases, boa_thresholds)))
@pytest.mark.filterwarnings(
    "ignore:User-specified a background spectral grid is overridden by atmosphere spectral grid"
)
def test_rami4atm_boa(mode_ckd_double, case, threshold, artefact_dir, plot_figures):
    spp = 100
    ctor = rami4atm.registry[case]
    case = case.replace("brfpp", "boa")
    srf = fresolver.load_dataset("srf/sentinel_2a-msi-4.nc")

    if "hom00" in case:
        extra_objects = {
            "boa_white_reference_patch": {
                "factory": "shape",
                "type": "rectangle",
                "center": [0, 0, 0.01],
                "edges": [1, 1],
                "bsdf": {"type": "lambertian", "reflectance": 1.0},
            }
        }
        target = [0, 0, 0.01]
    else:
        extra_objects = {
            "boa_white_reference_patch": {
                "factory": "shape",
                "type": "rectangle",
                "center": [0, 0, 2.025],
                "edges": [5, 5],
                "bsdf": {"type": "lambertian", "reflectance": 1.0},
            }
        }
        target = {
            "type": "rectangle",
            "xmin": -2.5,
            "xmax": 2.5,
            "ymin": -2.5,
            "ymax": 2.5,
            "z": 2.025,
        }

    exp1 = ctor(spp=spp)
    exp1 = attr.evolve(
        exp1,
        measures=[
            {
                "type": "mdistant",
                "spp": spp,
                "ray_offset": 0.05,
                "srf": srf,
                "construct": "hplane",
                "zeniths": np.arange(-75, 76, 1),
                "zeniths_units": "degree",
                "azimuth": 0.0,
                "azimuth_units": "degree",
                "target": target,
            }
        ],
    )
    exp2 = attr.evolve(exp1, extra_objects=extra_objects)
    dflux = {
        "type": "distantflux",
        "ray_offset": 0.05,
        "target": target,
        "srf": srf,
        "spp": spp,
    }
    exp3 = attr.evolve(exp1, measures=[dflux])
    exp4 = attr.evolve(exp2, measures=[dflux])

    result1 = eradiate.run(exp1)
    result2 = eradiate.run(exp2)
    result3 = eradiate.run(exp3)
    result4 = eradiate.run(exp4)

    result_hdrf = result1.radiance_srf / result2.radiance_srf
    result_bhr = result3.radiosity_srf / result4.radiosity_srf

    # Combine a reference dataset
    result = result1.rename(
        {"radiance_srf": "radiance_srf1", "radiance_var": "radiance_var1"}
    )
    result["radiance_srf2"] = result2.radiance_srf
    result["radiance_var2"] = result2.radiance_var
    result["radiosity_srf3"] = result3.radiosity_srf
    result["radiosity_var3"] = result3.sector_radiosity_var
    result["radiosity_srf4"] = result4.radiosity_srf
    result["radiosity_var4"] = result4.sector_radiosity_var
    result["hdrf"] = result_hdrf
    result["bhr"] = result_bhr

    # Handle numerical variance estimates, assuming:
    #  - Gaussian nature
    #  - Null covariance of different experiments or pixels
    #  - Local linearity of radiance and radiosity
    #  - Scalar SRF
    srf_weight = srf.srf.interp(w=result.w.values).fillna(1e-9)
    srf_sum = srf_weight.sum()

    # HDRF var
    S1 = result.radiance_srf1.sum() / srf_sum
    S2 = result.radiance_srf2.sum() / srf_sum
    v_S1 = (result.radiance_var1 * srf_weight**2).sum(dim="w") / (srf_weight**2).sum()
    v_S2 = (result.radiance_var2 * srf_weight**2).sum(dim="w") / (srf_weight**2).sum()

    d_hdrf_d_S1 = 1 / S2
    d_hdrf_d_S2 = -S1 / S2**2

    v_hdrf = d_hdrf_d_S1**2 * v_S1 + d_hdrf_d_S2**2 * v_S2
    result["hdrf_var"] = v_hdrf

    # BHR var
    S3 = result.radiosity_srf3.sum() / srf_sum
    S4 = result.radiosity_srf4.sum() / srf_sum
    v_S3 = (result.radiosity_var3 * srf_weight**2).sum(dim="w") / (srf_weight**2).sum()
    v_S4 = (result.radiosity_var4 * srf_weight**2).sum(dim="w") / (srf_weight**2).sum()

    d_bhr_d_S3 = 1 / S4
    d_bhr_d_S4 = -S3 / S4**2

    v_bhr = d_bhr_d_S3**2 * v_S3 + d_bhr_d_S4**2 * v_S4
    result["bhr_var"] = v_bhr.sum(dim=["x_index", "y_index"])

    logger.info(result._repr_html_(), html=True)

    test_name = case.replace("brfpp", "boa")
    reference = fresolver.load_dataset(
        f"tests/regression_test_references/rami4atm/{test_name}-ref.nc"
    )

    if plot_figures:
        fig = plt.figure()
        plt.grid()
        for K in range(1, 4):
            plt.fill_between(
                result.vza.squeeze(),
                reference.hdrf.squeeze() - K * np.sqrt(result.hdrf_var.squeeze()),
                reference.hdrf.squeeze() + K * np.sqrt(result.hdrf_var.squeeze()),
                alpha=0.3 / K,
                color="red",
                label=f"$HDRF_{{ref}}\\pm{K}\\sigma$",
            )
        plt.plot(
            result.vza.squeeze(),
            result.hdrf.squeeze(),
            # 3 * np.sqrt(result.hdrf_var.squeeze()),
            label="HDRF $\\pm 3\\sigma$",
            linewidth=1.0,
            # capsize=1,
        )
        plt.errorbar(
            0.0,
            result.bhr.squeeze(),
            3 * np.sqrt(result.bhr_var.squeeze()),
            marker="x",
            color="black",
            label="$BHR \\pm 3\\sigma$",
            linewidth=1.0,
        )
        plt.errorbar(
            0.0,
            reference.bhr.squeeze(),
            3 * np.sqrt(reference.bhr_var.squeeze()),
            marker="x",
            color="black",
            label="$BHR_{{ref}} \\pm 3\\sigma$",
            linewidth=1.0,
        )
        plt.xlabel("vza")
        plt.ylabel("reflectance")
        plt.legend()
        html_fig = figure_to_html(fig)
        logger.info(html_fig, html=True)

    test = RMSETest(
        name=test_name,
        value=result,
        reference=reference,
        threshold=threshold,
        archive_dir=artefact_dir,
        variable="hdrf",
        plot=plot_figures,
    )

    assert np.allclose(result.bhr.values, reference.bhr.values, atol=1e-4)
    assert test.run(plot_figures)
