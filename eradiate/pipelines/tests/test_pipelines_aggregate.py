import numpy as np

from eradiate.pipelines._aggregate import AggregateCKDQuad, AggregateSampleCount
from eradiate.pipelines._gather import Gather
from eradiate.units import symbol
from eradiate.units import unit_context_kernel as uck


def test_pipeline_step_aggregate_sample_count(results_mono_spp):
    # Initialise test data
    step = Gather(sensor_dims=("spp",))
    values = step.transform(results_mono_spp[0])

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


def test_pipeline_step_aggregate_ckd(results_ckd):
    # Initialise test data
    raw_results, exp = results_ckd
    step = Gather(
        sensor_dims=[], var=("radiance", {"units": symbol(uck.get("radiance"))})
    )
    values = step.transform(raw_results)

    # Configure step
    step = AggregateCKDQuad(measure=exp.measures[0], var="radiance")
    result = step.transform(values)

    # Dimension and variable checks
    assert "index" not in result.dims
    assert "bin" in result.dims
    assert "w" in result.coords
    assert "spp" in result.data_vars
    assert result.w.dims == ("bin",)

    # In the present case, the quadrature evaluates to 2/π
    assert np.allclose(2.0 / np.pi, result["radiance"].values)
    # Metadata of the variable for which aggregation is performed are copied
    assert result["radiance"].attrs == values["radiance"].attrs
    # Sample counts are averaged
    assert result.spp == 250
