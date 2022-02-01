import numpy as np
import pytest

import eradiate
from eradiate._mode import ModeFlags
from eradiate.experiments import OneDimExperiment
from eradiate.pipelines import ApplySpectralResponseFunction
from eradiate.scenes.measure import MultiDistantMeasure


@pytest.mark.parametrize(
    "mode_id, spectral_cfg",
    (
        ("mono", {"wavelengths": 550.0}),
        ("ckd", {"bin_set": "10nm", "bins": ["550"]}),
        ("ckd", {"bin_set": "10nm", "bins": ["550", "560"]}),
    ),
    ids=("mono", "ckd_single_bin", "ckd_multiple_bins"),
)
def test_apply_spectral_response_function_transform(mode_id, spectral_cfg):
    """
    Unit tests for ApplySpectralResponseFunction.transform().
    """
    # Prepare basic data
    eradiate.set_mode(mode_id)

    exp = OneDimExperiment(
        atmosphere=None,
        measures=MultiDistantMeasure.from_viewing_angles(
            id="measure",
            zeniths=[-60, -45, 0, 45, 60],
            azimuths=0.0,
            spp=1,
            spectral_cfg=spectral_cfg,
        ),
    )
    measure = exp.measures[0]
    exp.process(measure)

    # Apply first steps of post-processing
    pipeline = exp.pipeline(measure)
    values = pipeline.transform(measure.results, stop_after="add_viewing_angles")

    # Apply tested pipeline step
    step = ApplySpectralResponseFunction(measure=measure, vars=["radiance"])

    if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
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
