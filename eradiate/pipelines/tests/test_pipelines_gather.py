import numpy as np

from eradiate.pipelines._gather import Gather


def test_pipeline_step_gather_mono_ms_spp(results_mono_ms_spp):
    values = results_mono_ms_spp[0]

    # Configure this pipeline step according to the used sensor
    step = Gather(sensor_dims=("ms", "spp"))
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
    values = results_ckd_ms_spp[0]

    # Configure this pipeline step according to the used sensor
    step = Gather(sensor_dims=("ms", "spp"))
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
    values = results_mono_spp[0]

    # Configure this pipeline step according to the used sensor
    step = Gather(sensor_dims=("spp",))
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
    values = results_mono[0]

    # Configure this pipeline step according to the used sensor
    step = Gather()
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
