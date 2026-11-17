import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.grid import CartesianGrid


def run_experiment(components, merge_policy):
    exp = eradiate.experiments.AtmosphereExperiment(
        surface={"type": "lambertian", "reflectance": 0.3},
        atmosphere={
            "type": "heterogeneous",
            "components": components,
            "merge_policy": merge_policy,
        },
        illumination={"type": "directional", "zenith": 30.0, "azimuth": 0.0},
        measures={
            "type": "mdistant",
            "construct": "hplane",
            "azimuth": 0.0,
            "zeniths": np.arange(-60, 61, 20),
            "srf": {"type": "delta", "wavelengths": 550.0 * ureg.nm},
            "spp": 100000,
        },
    )
    return eradiate.run(exp)["brf"].squeeze().values


def particle_layer():
    return {"type": "particle_ot", "tau_ref": 0.5, "bottom": 0.0, "top": 5.0 * ureg.km}


def cloud():
    grid = CartesianGrid(
        x_edges=np.linspace(-5.0, 5.0, 3) * ureg.km,
        y_edges=np.linspace(-5.0, 5.0, 3) * ureg.km,
        z_edges=np.linspace(1.0, 3.0, 5) * ureg.km,
    )
    return {
        "type": "particle_ot",
        "tau_ref": 2.0,
        "bottom": 1.0 * ureg.km,
        "top": 3.0 * ureg.km,
        "grid": grid,
        "crop_to_grid": "xyz",
    }


@pytest.mark.slow
def test_heterogeneous_atmosphere_overlap(
    mode_mono_double, atmosphere_us_standard_mono
):
    """
    Overlapping media
    =================

    Merging components into a single medium and rendering them as overlapping
    media describe the same scene: results must agree within Monte Carlo noise.

    * Molecular and particle components on the same grid are merged or
      rendered as two overlapping media.
    * A cropped 3D cloud is added on its own grid: rendering all components as
      overlapping media matches merging the components which share a grid.
    """
    components = [atmosphere_us_standard_mono, particle_layer()]
    merged = run_experiment(components, "merge_all")
    overlapping = run_experiment(components, "overlap_all")
    np.testing.assert_allclose(overlapping, merged, rtol=0.02)

    components = [atmosphere_us_standard_mono, particle_layer(), cloud()]
    by_grid = run_experiment(components, "merge_by_grid")
    overlapping = run_experiment(components, "overlap_all")
    np.testing.assert_allclose(overlapping, by_grid, rtol=0.02)
