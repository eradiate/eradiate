import os

import attrs
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import eradiate
from eradiate import unit_registry as ureg


def compute(l="distant", sigma=1.0, rho=1.0, exp_cls="AtmosphereExperiment"):
    result = []
    spps = [4**i for i in range(0, 11)]

    for spp in spps:
        exp = getattr(eradiate.experiments, exp_cls)(
            geometry={"type": "plane_parallel", "toa_altitude": 1.0 * ureg.m},
            atmosphere={
                "type": "homogeneous",
                "sigma_a": sigma * ureg("m^-1"),
                "sigma_s": 0.0,
            },
            illumination={"type": "directional", "irradiance": 1.0},
            surface={"type": "lambertian", "reflectance": rho},
            measures={
                "type": "mdistant",
                "construct": "hplane",
                "zeniths": 0.0,
                "azimuth": 0.0,
                "ray_offset": None,
                "spp": spp,
            },
        )

        if l != "distant":
            exp = attrs.evolve(
                exp,
                measures={
                    "type": "mdistant",
                    "construct": "hplane",
                    "zeniths": 0.0,
                    "azimuth": 0.0,
                    "ray_offset": l * ureg.m,
                    "spp": spp,
                },
            )

        result.append(eradiate.run(exp).radiance.squeeze(drop=True))

    return xr.concat(result, dim="spp").assign_coords(spp=spps)


@pytest.mark.parametrize(
    "exp_cls", ["AtmosphereExperiment", "CanopyAtmosphereExperiment"]
)
def test_mdistant_insitu(artefact_dir, mode_mono, exp_cls):
    r"""
    In-situ multi-distant sensor
    ============================

    This test checks that the in-situ multi-distant sensor feature is correctly
    implemented. The test is performed for the following experiment classes:

    - ``AtmosphereExperiment``
    - ``CanopyAtmosphereExperiment``

    Rationale
    ---------

    - Surface: a Lambertian surface reflectance :math:`\rho = 1`.
    - Atmosphere: a homogeneous atmosphere with no scattering, an absorption
      coefficient :math:`\sigma = 1` and a thickness equal to 1.
    - Illumination: a directional light source at the zenith with radiance
      :math:`L_\mathrm{i} = 1.0`.
    - Sensor: an ``mdistant`` sensor targeting (0, 0, 0) in a single direction
      [0, 0, -1]. The ray offset is set to distant and a number of values
      between 0 and 1.

    Expected behaviour
    ------------------

    The recorded radiance is expected to be equal to
    :math:`\frac{1}{\pi} \exp \left( - \sigma (l + \mathrm{offset}) \right)`.
    """

    sigma = 1.0
    computed = {}
    expected = {}
    ls = ["distant", 0.99, 0.5, 0.01]

    for l in ls:
        computed[l] = compute(l, sigma=sigma, exp_cls=exp_cls)
        expected[l] = np.exp(-sigma * (1.0 + (l if l != "distant" else 1.0))) / np.pi

    # Plot results
    plt.figure()
    for l in ls:
        computed[l].plot(ls=":", marker=".", xscale="log", label=f"{l}")
        plt.axhline(expected[l], zorder=0, c="whitesmoke", ls="--")

    plt.legend(ncol=2)
    test_name = f"test_mdistant_insitu-{exp_cls}"
    plt.title(test_name)

    outdir = os.path.join(artefact_dir, "plots")
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.join(outdir, f"{test_name}.png")
    plt.savefig(filename, dpi=200, bbox_inches="tight")

    plt.close()

    # Actual test
    result = np.squeeze([computed[l].isel(spp=-1) for l in ls])
    expected = np.squeeze([expected[l] for l in ls])

    assert np.allclose(
        result, expected, rtol=1e-2
    ), f"{result = }, {expected = }\nPlot file: '{filename}'"
