import pytest

import eradiate
from eradiate.units import unit_registry as ureg


@pytest.fixture(scope="module")
def results_mono():
    # Single-sensor setup, no sample count-split
    eradiate.set_mode("mono_double")
    exp = eradiate.experiments.AtmosphereExperiment(
        atmosphere=None,
        surface={"type": "lambertian", "reflectance": 1.0},
        illumination={"type": "directional", "irradiance": 2.0},
        measures=[
            {
                "type": "hemispherical_distant",
                "film_resolution": (32, 32),
                "spp": 250,
                "srf": {"type": "multi_delta", "wavelengths": 550.0 * ureg.nm},
            }
        ],
    )
    exp.process()
    return exp.measures[0].mi_results, exp


@pytest.fixture(scope="module")
def results_mono_spp():
    # Single-sensor, sample count-split setup
    eradiate.set_mode("mono_double")
    exp = eradiate.experiments.AtmosphereExperiment(
        atmosphere=None,
        surface={"type": "lambertian", "reflectance": 1.0},
        illumination={"type": "directional", "irradiance": 2.0},
        measures=[
            {
                "type": "hemispherical_distant",
                "film_resolution": (32, 32),
                "spp": 250,
                "split_spp": 100,
                "srf": {"type": "multi_delta", "wavelengths": 550.0 * ureg.nm},
            }
        ],
    )
    exp.process()
    return exp.measures[0].results, exp


@pytest.fixture(scope="module")
def results_ckd():
    # Single-sensor setup, no sample count-split, CKD mode
    eradiate.set_mode("ckd_double")
    exp = eradiate.experiments.AtmosphereExperiment(
        atmosphere=None,
        surface={"type": "lambertian", "reflectance": 1.0},
        illumination={"type": "directional", "irradiance": 2.0},
        measures=[
            {
                "type": "hemispherical_distant",
                "film_resolution": (32, 32),
                "spp": 250,
                "srf": {"type": "multi_delta", "wavelengths": 550.0 * ureg.nm},
            }
        ],
    )
    exp.process()
    return exp.measures[0].mi_results, exp


@pytest.fixture(scope="module")
def results_ckd_spp():
    # Single-sensor, sample count-split setup
    eradiate.set_mode("ckd_double")
    exp = eradiate.experiments.AtmosphereExperiment(
        atmosphere=None,
        surface={"type": "lambertian", "reflectance": 1.0},
        illumination={"type": "directional", "irradiance": 2.0},
        measures=[
            {
                "type": "hemispherical_distant",
                "film_resolution": (32, 32),
                "spp": 250,
                "split_spp": 100,
                "srf": {"type": "multi_delta", "wavelengths": 550.0 * ureg.nm},
            }
        ],
    )
    exp.process()
    return exp.measures[0].results, exp
