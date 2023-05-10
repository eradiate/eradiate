import mitsuba as mi
import numpy as np

from eradiate.pipelines import Gather


def test_pipeline_step_gather_mono(results_mono):
    """
    Basic test case for :class:`.Gather` (monochromatic mode).
    """
    values: mi.Bitmap = results_mono[0]

    # Configure this pipeline step according to the used sensor
    var = "radiance"
    step = Gather(var=var)
    result = step.transform(values)

    # Check that dimensions are those we expect
    spectral_dims = {"w": 1}
    film_dims = {
        "y_index": 32,
        "x_index": 32,
    }  # Note: the 'channel' film dimension is dropped in mono mode
    all_dims = {**spectral_dims, **film_dims}
    assert dict(result.dims) == all_dims

    # Check that data vars are those we expected
    assert set(result.keys()) == {"spp", var}
    # Sensor data has all dimensions
    assert set(result[var].dims) == set(all_dims.keys())

    # Check radiance values
    assert np.allclose(result[var].values, 2.0 / np.pi)


def test_pipeline_step_gather_ckd(results_ckd):
    """
    Basic test case for :class:`.Gather` (CKD).
    """
    values = results_ckd[0]

    # Configure this pipeline step according to the used sensor
    var = "radiance"
    step = Gather(var=var)
    result = step.transform(values)

    # Check that dimensions are those we expect
    spectral_dims = {"bin": 1, "index": 2}  # 2 is the default value
    film_dims = {
        "y_index": 32,
        "x_index": 32,
    }  # Note: the 'channel' film dimension is dropped in mono mode
    all_dims = {**spectral_dims, **film_dims}
    assert dict(result.dims) == all_dims

    # Check that data vars are those we expected
    assert set(result.keys()) == {"spp", var}
    # Sensor data has all dimensions
    assert set(result[var].dims) == set(all_dims.keys())

    # Check radiance and sample count values
    assert np.allclose(result[var].values, 2.0 / np.pi)
