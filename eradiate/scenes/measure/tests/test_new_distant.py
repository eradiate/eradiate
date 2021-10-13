import eradiate
from eradiate.scenes.measure._new_distant import Assemble


def test_pipeline_step_assemble_mono_ms_spp(mode_mono):
    # Generate some test data
    # We use a multi-sensor, sample count-split setup
    exp = eradiate.experiments.OneDimExperiment(
        measures=[
            {
                "type": "distant_array",
                "directions": [[0, 0, -1], [0, -0.7, -0.7]],
                "spp": 250,
                "spectral_cfg": {"wavelengths": [500.0]},
            }
        ]
    )
    exp.measures[0]._spp_splitting_threshold = 100
    exp.process()
    values = exp.measures[0].results.raw

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


def test_pipeline_step_assemble_ckd_ms_spp(mode_ckd):
    # Generate some test data
    # We use a multi-sensor, sample count-split setup
    exp = eradiate.experiments.OneDimExperiment(
        measures=[
            {
                "type": "distant_array",
                "directions": [[0, 0, -1], [0, -0.7, -0.7]],
                "spp": 250,
                "spectral_cfg": {"bins": ["500"]},
            }
        ]
    )
    exp.measures[0]._spp_splitting_threshold = 100
    exp.process()
    values = exp.measures[0].results.raw

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


def test_pipeline_step_assemble_mono_spp(mode_mono):
    # Generate some test data
    # We use a sample count-split setup
    exp = eradiate.experiments.OneDimExperiment(
        measures=[
            {
                "type": "distant",
                "film_resolution": (32, 32),
                "spp": 250,
                "spectral_cfg": {"wavelengths": [500.0]},
            }
        ]
    )
    exp.measures[0]._spp_splitting_threshold = 100
    exp.process()
    values = exp.measures[0].results.raw

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


def test_pipeline_step_assemble_mono(mode_mono):
    # Generate some test data
    # We use a sample count-split setup
    exp = eradiate.experiments.OneDimExperiment(
        measures=[
            {
                "type": "distant",
                "film_resolution": (32, 32),
                "spp": 250,
                "spectral_cfg": {"wavelengths": [500.0]},
            }
        ]
    )
    exp.process()
    values = exp.measures[0].results.raw

    # Configure this pipeline step according to the used sensor
    step = Assemble(sensor_dims=())
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
