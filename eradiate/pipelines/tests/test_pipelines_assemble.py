import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate._mode import ModeFlags
from eradiate.experiments import OneDimExperiment
from eradiate.pipelines._aggregate import AggregateCKDQuad
from eradiate.pipelines._assemble import (
    AddIllumination,
    AddViewingAngles,
    _remap_viewing_angles_plane,
)
from eradiate.pipelines._core import Pipeline
from eradiate.pipelines._gather import Gather
from eradiate.scenes.measure import MultiDistantMeasure
from eradiate.scenes.measure._hemispherical_distant import HemisphericalDistantMeasure


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
    "measure_type, expected_zenith, expected_azimuth",
    (
        ("multi_distant-nohplane", [60, 45, 0, 45, 60], [180, 180, 0, 0, 0]),
        ("multi_distant-hplane", [-60, -45, 0, 45, 60], [0, 0, 0, 0, 0]),
        (
            "hemispherical_distant",
            [
                [41.409622, 41.409622],
                [41.409622, 41.409622],
            ],
            [
                [225, 135],
                [315, 45],
            ],
        ),
    ),
    ids=(
        "multi_distant-nohplane",
        "multi_distant-hplane",
        "hemispherical_distant",
    ),
)
def test_pipelines_add_viewing_angles(
    mode_mono, measure_type, expected_zenith, expected_azimuth
):
    # Initialise test data
    if measure_type == "multi_distant-nohplane":
        measure = MultiDistantMeasure.from_viewing_angles(
            zeniths=[-60, -45, 0, 45, 60],
            azimuths=0.0,
            auto_hplane=False,
            spp=1,
        )

    elif measure_type == "multi_distant-hplane":
        measure = MultiDistantMeasure.from_viewing_angles(
            zeniths=[-60, -45, 0, 45, 60],
            azimuths=0.0,
            auto_hplane=True,
            spp=1,
        )

    elif measure_type == "hemispherical_distant":
        measure = HemisphericalDistantMeasure(
            film_resolution=(2, 2),
            spp=1,
        )

    else:
        assert False

    exp = OneDimExperiment(atmosphere=None, measures=measure)
    exp.process()
    measure = exp.measures[0]

    # Apply basic post-processing
    values = Gather(var="lo").transform(measure.results)

    step = AddViewingAngles(measure=measure)
    result = step.transform(values)

    # Produced dataset has viewing angle coordinates
    assert "vza" in result.coords
    assert "vaa" in result.coords

    # Viewing angles are set to appropriate values
    assert np.allclose(expected_zenith, result.coords["vza"].values.squeeze())
    assert np.allclose(expected_azimuth, result.coords["vaa"].values.squeeze())


@pytest.mark.parametrize(
    "illumination_type, expected_dims",
    [
        ("directional", ["sza", "saa"]),
        ("constant", []),
    ],
    ids=["directional", "constant"],
)
def test_pipelines_add_illumination(modes_all_single, illumination_type, expected_dims):
    # Initialise test data
    if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
        spectral_cfg = {"wavelengths": [540.0, 550.0, 560.0]}
    elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
        spectral_cfg = {"bins": ["540", "550", "560"]}
    else:
        pytest.skip(f"unsupported mode {eradiate.mode().id}")

    exp = OneDimExperiment(
        atmosphere=None,
        illumination={"type": illumination_type},
        measures=MultiDistantMeasure(spectral_cfg=spectral_cfg),
    )
    exp.process()
    measure = exp.measures[0]

    # Apply basic post-processing
    values = Pipeline(
        steps=[
            ("gather", Gather(var="lo")),
            ("aggregate_ckd_quad", AggregateCKDQuad(var="lo", measure=measure)),
        ]
    ).transform(measure.results)

    step = AddIllumination(illumination=exp.illumination, measure=measure)
    result = step.transform(values)
    assert all(dim in result.dims for dim in expected_dims)
    assert "irradiance" in result.data_vars
