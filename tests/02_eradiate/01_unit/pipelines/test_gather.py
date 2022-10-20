import numpy as np

from eradiate.pipelines._gather import Gather


def test_pipeline_step_assemble_mono(results_mono):
    """
    Most basic test case for :class:`.Gather`:

    * mono mode;
    * single-sensor measure;
    * no sample count split.
    """
    values = results_mono[0]

    # Configure this pipeline step according to the used sensor
    var = "radiance"
    step = Gather(var=var)
    result = step.transform(values)

    # Check that dimensions are those we expect
    sensor_dims = {"spp_index": 1}
    spectral_dims = {"w": 1}
    film_dims = {
        "y_index": 32,
        "x_index": 32,
    }  # Note: the 'channel' film dimension is dropped in mono mode
    all_dims = {**sensor_dims, **spectral_dims, **film_dims}
    assert dict(result.dims) == all_dims

    # Check that data vars are those we expected
    assert set(result.keys()) == {var, "spp"}
    # Sensor data has all dimensions
    assert set(result[var].dims) == set(all_dims.keys())
    # SPP data has only sensor and spectral dimensions
    assert set(result["spp"].dims) == set(
        list(sensor_dims.keys()) + list(spectral_dims.keys())
    )

    # Check radiance and sample count values
    assert np.allclose(result[var].values, 2.0 / np.pi)
    assert np.all([250] == result.spp.values)


def test_pipeline_step_assemble_ckd(results_ckd):
    """
    Most basic test case for :class:`.Gather`:

    * ckd mode;
    * single-sensor measure;
    * no sample count split.
    """
    values = results_ckd[0]

    # Configure this pipeline step according to the used sensor
    var = "radiance"
    step = Gather(var=var)
    result = step.transform(values)

    # Check that dimensions are those we expect
    sensor_dims = {"spp_index": 1}
    spectral_dims = {"bin": 1, "index": 16}
    film_dims = {
        "y_index": 32,
        "x_index": 32,
    }  # Note: the 'channel' film dimension is dropped in mono mode
    all_dims = {**sensor_dims, **spectral_dims, **film_dims}
    assert dict(result.dims) == all_dims

    # Check that data vars are those we expected
    assert set(result.keys()) == {var, "spp"}
    # Sensor data has all dimensions
    assert set(result[var].dims) == set(all_dims.keys())
    # SPP data has only sensor and spectral dimensions
    assert set(result["spp"].dims) == set(
        list(sensor_dims.keys()) + list(spectral_dims.keys())
    )

    # Check radiance and sample count values
    assert np.allclose(result[var].values, 2.0 / np.pi)
    assert np.all([250] == result.spp.values)


def test_pipeline_step_assemble_mono_spp(results_mono_spp):
    """
    Test case for :class:`.Gather` with

    * mono mode;
    * single-sensor measure;
    * sample count split.
    """
    values = results_mono_spp[0]

    # Configure this pipeline step according to the used sensor
    var = "radiance"
    step = Gather(var=var, sensor_dims=("spp",))
    result = step.transform(values)

    # Check that all dimensions are here as expected
    sensor_dims = {"spp_index": 3}
    spectral_dims = {"w": 1}
    film_dims = {
        "y_index": 32,
        "x_index": 32,
    }  # Note: the 'channel' film dimension is dropped in mono mode
    all_dims = {**sensor_dims, **spectral_dims, **film_dims}

    # Check that data vars are those we expected
    assert set(result.keys()) == {var, "spp"}
    # Sensor data has all dimensions
    assert set(result[var].dims) == set(all_dims.keys())
    # SPP data has only sensor and spectral dimensions
    assert set(result["spp"].dims) == set(
        list(sensor_dims.keys()) + list(spectral_dims.keys())
    )

    # Check radiance and sample count values
    assert np.allclose(result[var].values, 2.0 / np.pi)
    assert np.all([100, 100, 50] == result.spp.values)


def test_pipeline_step_assemble_ckd_spp(results_ckd_spp):
    """
    Test case for :class:`.Gather` with

    * mono mode;
    * single-sensor measure;
    * sample count split.
    """
    values = results_ckd_spp[0]

    # Configure this pipeline step according to the used sensor
    var = "radiance"
    step = Gather(var=var, sensor_dims=("spp",))
    result = step.transform(values)

    # Check that all dimensions are here as expected
    sensor_dims = {"spp_index": 3}
    spectral_dims = {"bin": 1, "index": 16}
    film_dims = {
        "y_index": 32,
        "x_index": 32,
    }  # Note: the 'channel' film dimension is dropped in mono mode
    all_dims = {**sensor_dims, **spectral_dims, **film_dims}

    # Check that data vars are those we expected
    assert set(result.keys()) == {var, "spp"}
    # Sensor data has all dimensions
    assert set(result[var].dims) == set(all_dims.keys())
    # SPP data has only sensor and spectral dimensions
    assert set(result["spp"].dims) == set(
        list(sensor_dims.keys()) + list(spectral_dims.keys())
    )

    # Check radiance and sample count values
    assert np.allclose(result[var].values, 2.0 / np.pi)
    assert np.all([100, 100, 50] == result.spp.values)


# Multi-sensor tests are removed until a multi-sensor measure is again available

# def test_pipeline_step_gather_mono_ms_spp(results_mono_ms_spp):
#     values = results_mono_ms_spp[0]
#
#     # Configure this pipeline step according to the used sensor
#     step = Gather(sensor_dims=("ms", "spp"))
#     result = step.transform(values)
#
#     # Check that all dimensions are here as expected
#     assert result.dims == {
#         "w": 1,
#         "ms_index": 2,
#         "spp_index": 3,
#         "y_index": 1,
#         "x_index": 1,
#         # "channel": 1,  # channel dimension should be dropped in modes using mono variants
#     }
#     assert set(result.keys()) == {"img", "spp"}
#
#     # Check that variables have appropriate dimensions
#     assert result.img.dims == {
#         "w": 1,
#         "ms_index": 2,
#         "spp_index": 3,
#         "y_index": 1,
#         "x_index": 1,
#     }
#     assert result.spp.dims == {"spp_index", "w"}
#
#     # Check radiance and sample count values
#     assert np.allclose(2.0 / np.pi, result.img.values)
#     assert np.all([100, 100, 50] == result.spp.values)
#
#
# def test_pipeline_step_assemble_ckd_ms_spp(results_ckd_ms_spp):
#     values = results_ckd_ms_spp[0]
#
#     # Configure this pipeline step according to the used sensor
#     step = Gather(sensor_dims=("ms", "spp"))
#     result = step.transform(values)
#
#     # Check that all dimensions are here as expected
#     assert result.dims == {
#         "bin": 1,
#         "index": 16,
#         "ms_index": 2,
#         "spp_index": 3,
#         "y_index": 1,
#         "x_index": 1,
#         "channel": 1,
#     }
#     assert set(result.keys()) == {"img", "spp"}
#
#     # Check radiance and sample count values
#     assert np.allclose(2.0 / np.pi, result.img.values)
#     assert np.all([100, 100, 50] == result.spp.values)
