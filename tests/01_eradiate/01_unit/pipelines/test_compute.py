import numpy as np
import pytest

import eradiate
from eradiate.experiments import AtmosphereExperiment
from eradiate.pipelines import ApplySpectralResponseFunction
from eradiate.scenes.measure import MultiDistantMeasure
from eradiate.units import unit_registry as ureg


@pytest.mark.parametrize(
    "mode_id, srf",
    (
        ("mono", {"type": "multi_delta", "wavelengths": 550.0 * ureg.nm}),
        (
            "ckd",
            {
                "type": "interpolated",
                "wavelengths": [550, 560] * ureg.nm,
                "values": [1.0, 1.0] * ureg.dimensionless,
            },
        ),
    ),
    ids=("mono", "ckd_multiple_bins"),
)
def test_apply_spectral_response_function_transform(mode_id, srf):
    """
    Unit tests for ApplySpectralResponseFunction.transform().
    """
    # Prepare basic data
    eradiate.set_mode(mode_id)

    exp = AtmosphereExperiment(
        atmosphere=None,
        measures=MultiDistantMeasure.hplane(
            id="measure",
            zeniths=[-60, -45, 0, 45, 60],
            azimuth=0.0,
            spp=1,
            srf=srf,
        ),
    )
    measure = exp.measures[0]
    exp.process()

    # Apply first steps of post-processing
    pipeline = exp.pipeline(measure)
    values = pipeline.transform(measure.mi_results, stop_after="add_viewing_angles")

    # Apply tested pipeline step
    step = ApplySpectralResponseFunction(measure=measure, vars=["radiance"])

    if eradiate.mode().is_mono:
        # In mono modes, the dataset produced by previous steps is missing
        # bin data required for the step to run successfully
        with pytest.raises(ValueError):
            step.transform(values)
        return

    # In binned modes, computation goes through
    result = step.transform(values)

    # The step adds a SRF-weighted variable
    assert "radiance_srf" in result.data_vars
    assert np.all(result.radiance_srf > 0.0)
