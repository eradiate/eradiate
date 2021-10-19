import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.measure._new_distant import (
    AggregateSampleCount,
    Assemble,
    MultiDistantMeasure,
)


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


def test_pipeline_step_assemble_mono_ms_spp(results_mono_ms_spp):
    values = results_mono_ms_spp

    # Configure this pipeline step according to the used sensor
    step = Assemble(sensor_dims=("ms", "spp"))
    result = step.transform(values)

    # Check that all dimensions are here as expected
    assert result.dims == {
        "w": 1,
        "ms_index": 2,
        "spp_index": 3,
        "y_index": 1,
        "x_index": 1,
        "channel": 1,
    }
    assert set(result.keys()) == {"img", "spp"}

    # Check radiance and sample count values
    assert np.allclose(2.0 / np.pi, result.img.values)
    assert np.all([100, 100, 50] == result.spp.values)


def test_pipeline_step_assemble_ckd_ms_spp(results_ckd_ms_spp):
    values = results_ckd_ms_spp

    # Configure this pipeline step according to the used sensor
    step = Assemble(sensor_dims=("ms", "spp"))
    result = step.transform(values)

    # Check that all dimensions are here as expected
    assert result.dims == {
        "bin": 1,
        "index": 16,
        "ms_index": 2,
        "spp_index": 3,
        "y_index": 1,
        "x_index": 1,
        "channel": 1,
    }
    assert set(result.keys()) == {"img", "spp"}

    # Check radiance and sample count values
    assert np.allclose(2.0 / np.pi, result.img.values)
    assert np.all([100, 100, 50] == result.spp.values)


def test_pipeline_step_assemble_mono_spp(results_mono_spp):
    values = results_mono_spp

    # Configure this pipeline step according to the used sensor
    step = Assemble(sensor_dims=("spp",))
    result = step.transform(values)

    # Check that all dimensions are here as expected
    assert result.dims == {
        "w": 1,
        "spp_index": 3,
        "y_index": 32,
        "x_index": 32,
        "channel": 1,
    }
    assert set(result.keys()) == {"img", "spp"}

    # Check radiance and sample count values
    assert np.allclose(2.0 / np.pi, result.img.values)
    assert np.all([100, 100, 50] == result.spp.values)


def test_pipeline_step_assemble_mono(results_mono):
    values = results_mono

    # Configure this pipeline step according to the used sensor
    step = Assemble()
    result = step.transform(values)

    # Check that all dimensions are here as expected
    assert result.dims == {
        "w": 1,
        "spp_index": 1,
        "y_index": 32,
        "x_index": 32,
        "channel": 1,
    }
    assert set(result.keys()) == {"img", "spp"}

    # Check radiance and sample count values
    assert np.allclose(2.0 / np.pi, result.img.values)
    assert np.all([250] == result.spp.values)


def test_pipeline_step_aggregate_sample_count(results_mono_spp):
    # Initialise test data
    step = Assemble(sensor_dims=("spp",))
    values = step.transform(results_mono_spp)

    # Configure step
    step = AggregateSampleCount()
    result = step.transform(values)

    # spp dimension is not here
    assert "spp_index" not in result.coords
    # spp variable is still here
    assert "spp" in result.data_vars

    # Sample counts are summed
    assert result.spp == 250
    # Radiance values are averaged
    assert np.allclose(2.0 / np.pi, result.img.values)


def test_multi_distant_measure_construct(mode_mono):
    """Basic constructor testing for MultiDistantMeasure."""

    ctx = KernelDictContext()

    # Constructing without argument succeeds
    measure = MultiDistantMeasure()

    # The produced kernel dictionary can be instantiated
    kernel_dict = measure.kernel_dict(ctx)
    assert kernel_dict.load().expand()[0]


def test_multi_distant_measure_viewing_angles(mode_mono):
    # Viewing angle computation is correct
    measure = MultiDistantMeasure(
        directions=[
            [0, 0, -1],
            [-1, 0, -1],
            [0, -1, -1],
            [-1, -1, -1],
        ]
    )

    assert np.allclose(
        [
            [0, 0],
            [45, 0],
            [45, 90],
            [np.rad2deg(np.arccos(1 / np.sqrt(3))), 45],
        ]
        * ureg.deg,
        measure.viewing_angles,
    )


def test_multi_distant_measure_from_viewing_angles(mode_mono):
    # Constructing from viewing angles yields correct directions
    angles = [
        (0, 0),
        (45, 0),
        (90, 0),
        (45, 0),
        (45, 45),
        (45, 90),
        (90, 0),
        (90, 45),
        (90, 90),
    ]
    measure = MultiDistantMeasure.from_viewing_angles(angles)
    assert np.allclose(angles * ureg.deg, measure.viewing_angles)
