from pprint import pprint

import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate._mode import ModeFlags
from eradiate.exceptions import UnsupportedModeError
from eradiate.experiments import OneDimExperiment
from eradiate.pipelines._assemble import AddViewingAngles, _remap_viewing_angles_plane
from eradiate.pipelines._core import Pipeline
from eradiate.pipelines._gather import Gather
from eradiate.scenes.measure import MultiDistantMeasure


def test_remap_viewing_angles_plane():
    # Usage with "normal" parameters
    theta, phi = _remap_viewing_angles_plane(
        0 * ureg.deg,
        [60, 45, 0, 45, 60] * ureg.deg,
        [-180, -180, 0, 0, 0] * ureg.deg,
    )
    assert np.allclose(theta, [-60, -45, 0, 45, 60] * ureg.deg)
    assert np.allclose(phi, 0 * ureg.deg)

    # Usage with "exotic" parameters
    theta, phi = _remap_viewing_angles_plane(
        0 * ureg.deg,
        [60, 45, 0, 45, 60] * ureg.deg,
        [-180, 180, 90, 720, 0] * ureg.deg,
    )
    assert np.allclose(theta, [-60, -45, 0, 45, 60] * ureg.deg)
    assert np.allclose(phi, 0 * ureg.deg)

    # Improper direction ordering is detected
    with pytest.warns(Warning):
        theta, phi = _remap_viewing_angles_plane(
            180 * ureg.deg,
            [45, 0, 45] * ureg.deg,
            [-180, 0, 0] * ureg.deg,
        )
    assert np.allclose(theta, [45, 0, -45] * ureg.deg)
    assert np.allclose(phi, 180 * ureg.deg)


@pytest.mark.parametrize(
    "hplane, expected_zenith, expected_azimuth",
    (
        (False, [60, 45, 0, 45, 60], [180, 180, 0, 0, 0]),
        (True, [-60, -45, 0, 45, 60], [0, 0, 0, 0, 0]),
    ),
    ids=("no_hplane", "hplane"),
)
def test_multi_distant_measure_add_viewing_angles(
    mode_mono, hplane, expected_zenith, expected_azimuth
):
    # Initialise test data
    exp = OneDimExperiment(
        atmosphere=None,
        measures=MultiDistantMeasure.from_viewing_angles(
            zeniths=[-60, -45, 0, 45, 60],
            azimuths=0.0,
            auto_hplane=hplane,
            spp=1,
        ),
    )
    exp.process()
    measure = exp.measures[0]

    # Collect post-processing pipeline and apply steps prior to "add_viewing_angles"
    values = Gather(var="lo").transform(measure.results.raw)

    step = AddViewingAngles(measure=measure)
    result = step.transform(values)

    # Produced dataset has viewing angle coordinates
    assert "vza" in result.coords
    assert "vaa" in result.coords

    # Viewing angles are set to appropriate values
    assert np.allclose(expected_zenith, result.coords["vza"].values.ravel())
    assert np.allclose(expected_azimuth, result.coords["vaa"].values.ravel())


def test_pipelines_add_illumination(mode_ckd):
    # Initialise test data
    if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
        spectral_cfg = {"wavelengths": [540.0, 550.0, 560.0]}
    elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
        spectral_cfg = {"bins": ["540", "550", "560"]}
    else:
        raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    exp = OneDimExperiment(
        atmosphere=None,
        measures=MultiDistantMeasure(spectral_cfg=spectral_cfg),
    )
    exp.process()
    measure = exp.measures[0]
    pprint(measure)

    # Collect post-processing pipeline and apply steps prior to "add_viewing_angles"
    pipeline = measure.pipeline
    values = pipeline.transform(measure.results.raw, stop="add_viewing_angles")
    add_viewing_angles = pipeline.named_steps["add_viewing_angles"]
    result = add_viewing_angles.transform(values)
    pprint(result)
