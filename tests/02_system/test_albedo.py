import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eradiate
from eradiate.experiments import AtmosphereExperiment
from eradiate.units import unit_registry as ureg


@pytest.mark.slow
def test_albedo(mode_mono, artefact_dir):
    """
    Albedo
    ======

    This system test verifies the behaviour of experiments capable of albedo
    computation.

    Rationale
    ---------

    We use a scene consisting of a single surface with a diffuse, spectrally
    non-uniform BRDF.

    * Geometry: A Lambertian surface with linearly varying reflectance for
      0 to 1 between 500 and 700 nm.
    * Illumination: Directional illumination from the zenith (default irradiance)
      or constant illumination (default radiance).
    * Atmosphere/canopy: No atmosphere nor canopy.
    * Measure: Distant albedo measure with a film of size 64 x 64. This
      guarantees reasonable stratification of the film sampling and ensures
      quick converge to the expected value, thus allowing for a low sample
      count.

    The test is run for the ``AtmosphereExperiment`` and ``CanopyExperiment``
    classes.

    Expected behaviour
    ------------------

    We expect the albedo to be equal to the reflectance of the surface.

    Results
    -------

    .. image:: generated/plots/test_albedo_atmosphere_directional.png
       :width: 100%

    .. image:: generated/plots/test_albedo_canopy_directional.png
       :width: 100%

    .. image:: generated/plots/test_albedo_atmosphere_constant.png
       :width: 100%

    .. image:: generated/plots/test_albedo_canopy_constant.png
       :width: 100%

    """
    exps = {
        "atmosphere_directional": AtmosphereExperiment(
            measures=[
                {
                    "type": "distant_flux",
                    "srf": {
                        "type": "multi_delta",
                        "wavelengths": [500.0, 550.0, 600.0, 650.0, 700.0] * ureg.nm,
                    },
                    "film_resolution": (64, 64),
                    "spp": 256,
                }
            ],
            atmosphere=None,
            surface={
                "type": "lambertian",
                "reflectance": {
                    "type": "interpolated",
                    "wavelengths": [500.0, 700.0],
                    "values": [0.0, 1.0],
                },
            },
            illumination={"type": "directional", "zenith": 0.0},
        ),
        "atmosphere_constant": AtmosphereExperiment(
            measures=[
                {
                    "type": "distant_flux",
                    "srf": {
                        "type": "multi_delta",
                        "wavelengths": [500.0, 550.0, 600.0, 650.0, 700.0] * ureg.nm,
                    },
                    "film_resolution": (64, 64),
                    "spp": 256,
                }
            ],
            atmosphere=None,
            surface={
                "type": "lambertian",
                "reflectance": {
                    "type": "interpolated",
                    "wavelengths": [500.0, 700.0],
                    "values": [0.0, 1.0],
                },
            },
            illumination={"type": "constant"},
        ),
        "canopy_directional": eradiate.experiments.CanopyExperiment(
            measures=[
                {
                    "type": "distant_flux",
                    "srf": {
                        "type": "multi_delta",
                        "wavelengths": [500.0, 550.0, 600.0, 650.0, 700.0] * ureg.nm,
                    },
                    "film_resolution": (64, 64),
                    "spp": 256,
                }
            ],
            canopy=None,
            surface={
                "type": "lambertian",
                "reflectance": {
                    "type": "interpolated",
                    "wavelengths": [500.0, 700.0],
                    "values": [0.0, 1.0],
                },
            },
            illumination={"type": "directional", "zenith": 0.0},
        ),
        "canopy_constant": eradiate.experiments.CanopyExperiment(
            measures=[
                {
                    "type": "distant_flux",
                    "srf": {
                        "type": "multi_delta",
                        "wavelengths": [500.0, 550.0, 600.0, 650.0, 700.0] * ureg.nm,
                    },
                    "film_resolution": (64, 64),
                    "spp": 256,
                }
            ],
            canopy=None,
            surface={
                "type": "lambertian",
                "reflectance": {
                    "type": "interpolated",
                    "wavelengths": [500.0, 700.0],
                    "values": [0.0, 1.0],
                },
            },
            illumination={"type": "directional", "zenith": 0.0},
        ),
    }

    for exp_name, exp in exps.items():
        # Run simulation
        results = eradiate.run(exp)

        # Plot results
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
        wavelengths = results["albedo"].w.values
        albedos = results["albedo"].values.squeeze()
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        ax1.plot(wavelengths, albedos, linestyle="--", marker="o")
        ax1.set_title("Albedo")
        ax1.set_xlabel("Wavelength [nm]")

        rdiffs = np.divide(
            (albedos - expected),
            expected,
            where=expected != 0.0,
            out=np.zeros_like(expected),
        )
        ax2.plot(wavelengths, rdiffs, linestyle="--", marker="o")
        ax2.set_title("Relative difference")
        ax2.set_xlabel("Wavelength [nm]")
        rdiffs_max = np.max(np.abs(rdiffs[~np.isnan(rdiffs)]))
        exp = np.ceil(np.log10(rdiffs_max))
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim([-(10**exp), 10**exp])
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.ticklabel_format(axis="y", style="sci", scilimits=[-3, 3])
        # Hide offset label and add it as axis label
        if abs(exp) >= 3:
            ax2.yaxis.offsetText.set_visible(False)
            ax2.yaxis.set_label_text(f"×$10^{{{int(exp)}}}$")

        plt.suptitle(f"Case: {exp_name}")
        plt.tight_layout()

        filename = f"test_albedo_{exp_name}.png"
        outdir = os.path.join(artefact_dir, "plots")
        os.makedirs(outdir, exist_ok=True)
        fname_plot = os.path.join(outdir, filename)

        fig.savefig(fname_plot, dpi=200)
        plt.close()

        # Check results
        assert np.allclose(np.squeeze(results["albedo"].values), expected, atol=1e-3)
