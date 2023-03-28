import numpy as np

from eradiate.experiments import AtmosphereExperiment
from eradiate.pipelines import (
    AggregateCKDQuad,
    AggregateRadiosity,
    Gather,
)
from eradiate.units import symbol
from eradiate.units import unit_context_kernel as uck
from eradiate.units import unit_registry as ureg


def test_aggregate_ckd(results_ckd):
    # Initialise test data
    raw_results, exp = results_ckd
    step = Gather(var=("radiance", {"units": symbol(uck.get("radiance"))}))
    values = step.transform(raw_results)

    # Configure step
    step = AggregateCKDQuad(
        measure=exp.measures[0],
        var="radiance",
        binset=exp.spectral_set[0],
    )
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
                "srf": {"type": "multi_delta", "wavelengths": [550.0] * ureg.nm},
            }
        ],
    )
    exp.process()
    values = Gather(
        var=("sector_radiosity", {"units": symbol(uck.get("irradiance"))})
    ).transform(exp.measures[0].mi_results)

    # Configure and apply step
    step = AggregateRadiosity(
        sector_radiosity_var="sector_radiosity", radiosity_var="radiosity"
    )
    result = step.transform(values)

    # Check that radiosity dimensions are correct
    assert not {"x_index", "y_index"}.issubset(result["radiosity"].dims)
    # This setup conserves energy
    assert np.isclose(irradiance, result["radiosity"], rtol=1e-4)
