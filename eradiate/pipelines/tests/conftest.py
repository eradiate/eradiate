import pytest

import eradiate


@pytest.fixture(scope="module")
def results_mono_ms_spp():
    # Multi-sensor, sample count-split setup
    eradiate.set_mode("mono")
    exp = eradiate.experiments.OneDimExperiment(
        atmosphere=None,
        surface={"type": "lambertian", "reflectance": 1.0},
        illumination={"type": "directional", "irradiance": 2.0},
        measures=[
            {
                "type": "distant_array",
                "directions": [[0, 0, 1], [0, 0.7, 0.7]],
                "spp": 250,
                "spectral_cfg": {"wavelengths": [500.0]},
            }
        ],
    )
    exp.measures[0]._spp_splitting_threshold = 100
    exp.process()
    return exp.measures[0].results.raw


@pytest.fixture(scope="module")
def results_ckd_ms_spp():
    # Multi-sensor, sample count-split setup
    eradiate.set_mode("ckd")
    exp = eradiate.experiments.OneDimExperiment(
        atmosphere=None,
        surface={"type": "lambertian", "reflectance": 1.0},
        illumination={"type": "directional", "irradiance": 2.0},
        measures=[
            {
                "type": "distant_array",
                "directions": [[0, 0, 1], [0, 0.7, 0.7]],
                "spp": 250,
                "spectral_cfg": {"bins": ["500"]},
            }
        ],
    )
    exp.measures[0]._spp_splitting_threshold = 100
    exp.process()
    return exp.measures[0].results.raw


@pytest.fixture(scope="module")
def results_mono_spp():
    # Single-sensor, sample count-split setup
    eradiate.set_mode("mono")
    exp = eradiate.experiments.OneDimExperiment(
        atmosphere=None,
        surface={"type": "lambertian", "reflectance": 1.0},
        illumination={"type": "directional", "irradiance": 2.0},
        measures=[
            {
                "type": "distant",
                "film_resolution": (32, 32),
                "spp": 250,
                "spectral_cfg": {"wavelengths": [500.0]},
            }
        ],
    )
    exp.measures[0]._spp_splitting_threshold = 100
    exp.process()
    return exp.measures[0].results.raw


@pytest.fixture(scope="module")
def results_mono():
    # Single-sensor setup, no sample count-split
    eradiate.set_mode("mono")
    exp = eradiate.experiments.OneDimExperiment(
        atmosphere=None,
        surface={"type": "lambertian", "reflectance": 1.0},
        illumination={"type": "directional", "irradiance": 2.0},
        measures=[
            {
                "type": "distant",
                "film_resolution": (32, 32),
                "spp": 250,
                "spectral_cfg": {"wavelengths": [500.0]},
            }
        ],
    )
    exp.process()
    return exp.measures[0].results.raw
