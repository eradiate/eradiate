import numpy as np

from eradiate.experiments import AtmosphereExperiment
from eradiate.pipelines import AggregateRadiosity, GatherMono
from eradiate.units import symbol
from eradiate.units import unit_context_kernel as uck
from eradiate.units import unit_registry as ureg


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
    values = GatherMono(
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
