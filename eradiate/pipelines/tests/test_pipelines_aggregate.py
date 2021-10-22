import numpy as np

from eradiate.pipelines._aggregate import AggregateSampleCount
from eradiate.pipelines._gather import Gather


def test_pipeline_step_aggregate_sample_count(results_mono_spp):
    # Initialise test data
    step = Gather(sensor_dims=("spp",))
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
