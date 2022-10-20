import numpy as np

from eradiate.experiments import AtmosphereExperiment
from eradiate.pipelines._aggregate import (
    AggregateCKDQuad,
    AggregateRadiosity,
    AggregateSampleCount,
)
from eradiate.pipelines._gather import Gather
from eradiate.units import symbol
from eradiate.units import unit_context_kernel as uck


def test_aggregate_sample_count(results_mono_spp):
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
    # spp variable is indexed only on spectral coordinate
    assert set(result.spp.dims) == set(values.spp.dims) - {"spp_index"}

    # Sample counts are summed
    assert result.spp == 250
    # Radiance values are averaged
    assert np.allclose(result.img.values, 2.0 / np.pi)


def test_aggregate_ckd(results_ckd):
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
    assert "w" in result.dims
    assert "bin" not in result.dims
    assert "bin" in result.coords
    assert "bin_wmin" in result.coords
    assert "bin_wmax" in result.coords
    assert "spp" in result.data_vars
    assert result.bin.dims == ("w",)
    assert result.bin_wmin.dims == ("w",)
    assert result.bin_wmax.dims == ("w",)

    # In the present case, the quadrature evaluates to 2/π
    assert np.allclose(result["radiance"].values, 2.0 / np.pi)
    # Metadata of the variable for which aggregation is performed are copied
    assert result["radiance"].attrs == values["radiance"].attrs
    # Sample counts are averaged
    assert result.spp == 250


def test_aggregate_radiosity(mode_mono):
    # Initialise test data
    irradiance = 2.0

    exp = AtmosphereExperiment(
        atmosphere=None,
        surface={"type": "lambertian", "reflectance": 1.0},
        illumination={"type": "directional", "irradiance": irradiance},
        measures=[
            {
                "type": "distant_flux",
                "film_resolution": (32, 32),
                "spp": 1000,
                "spectral_cfg": {"wavelengths": [550.0]},
            }
        ],
    )
    exp.process()
    values = Gather(
        sensor_dims=[],
        var=("sector_radiosity", {"units": symbol(uck.get("irradiance"))}),
    ).transform(exp.measures[0].results)
    print(values)

    # Configure and apply step
    step = AggregateRadiosity(
        sector_radiosity_var="sector_radiosity", radiosity_var="radiosity"
    )
    result = step.transform(values)

    # Check that radiosity dimensions are correct
    print(result)
    assert not {"x_index", "y_index"}.issubset(result["radiosity"].dims)
    # This setup conserves energy
    assert np.isclose(irradiance, result["radiosity"], rtol=1e-4)
