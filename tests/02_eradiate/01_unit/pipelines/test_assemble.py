import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.experiments import AtmosphereExperiment
from eradiate.pipelines import (
    AddIllumination,
    AddSpectralResponseFunction,
    AddViewingAngles,
    AggregateCKDQuad,
    Gather,
    Pipeline,
)
from eradiate.pipelines._assemble import _remap_viewing_angles_plane
from eradiate.scenes.measure import HemisphericalDistantMeasure, MultiDistantMeasure
from eradiate.scenes.spectra import InterpolatedSpectrum, MultiDeltaSpectrum


@pytest.mark.parametrize(
    "illumination_type, expected_dims",
    [
        ("directional", ["sza", "saa"]),
        ("constant", []),
    ],
    ids=["directional", "constant"],
)
def test_add_illumination(modes_all_single, illumination_type, expected_dims):
    exp = AtmosphereExperiment(
        atmosphere=None,
        illumination={"type": illumination_type},
        measures=MultiDistantMeasure(
            srf=MultiDeltaSpectrum(wavelengths=[440.0, 550.0, 660.0] * ureg.nm)
        ),
    )
    exp.process()
    measure_index = 0
    measure = exp.measures[0]

    # Apply basic post-processing
    steps = [("gather", Gather(var="radiance"))]
    if eradiate.mode().is_ckd:
        steps.append(
            (
                "aggregate_ckd_quad",
                AggregateCKDQuad(
                    var="radiance",
                    measure=measure,
                    binset=exp.spectral_set[measure_index],
                ),
            )
        )

    values = Pipeline(steps=steps).transform(measure.mi_results)

    step = AddIllumination(illumination=exp.illumination, measure=measure)
    result = step.transform(values)
    # Irradiance is here and is indexed in Sun angles and spectral coordinate
    irradiance = result.data_vars["irradiance"]
    assert set(irradiance.dims) == {"w"}.union(set(expected_dims))


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
        (
            "multi_distant-aring",
            [45, 45, 45, 45, 45, 45, 45, 45],
            [0, 45, 90, 135, 180, 225, 270, 315],
        ),
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
        "multi_distant-aring",
        "multi_distant-hplane",
        "hemispherical_distant",
    ),
)
def test_add_viewing_angles(mode_mono, measure_type, expected_zenith, expected_azimuth):
    # Initialise test data
    if measure_type == "multi_distant-aring":
        measure = MultiDistantMeasure.aring(
            zenith=45,
            azimuths=np.linspace(0.0, 360.0, 8, endpoint=False),
            spp=1,
        )

    elif measure_type == "multi_distant-hplane":
        measure = MultiDistantMeasure.hplane(
            zeniths=[-60, -45, 0, 45, 60],
            azimuth=0.0,
            spp=1,
        )

    elif measure_type == "hemispherical_distant":
        measure = HemisphericalDistantMeasure(film_resolution=(2, 2), spp=1)

    else:
        assert False

    exp = AtmosphereExperiment(atmosphere=None, measures=measure)
    exp.process()
    measure_index = 0
    measure = exp.measures[measure_index]

    # Apply basic post-processing
    values = Gather(var="radiance").transform(measure.mi_results)

    step = AddViewingAngles(measure=measure)
    result = step.transform(values)

    # Produced dataset has viewing angle coordinates
    assert "vza" in result.coords
    assert "vaa" in result.coords

    # Viewing angles are set to appropriate values
    assert np.allclose(result.coords["vza"].values.squeeze(), expected_zenith)
    assert np.allclose(result.coords["vaa"].values.squeeze(), expected_azimuth)


def test_add_srf(modes_all_single):
    exp = AtmosphereExperiment(
        atmosphere=None,
        measures=MultiDistantMeasure.hplane(
            zeniths=[-60, -45, 0, 45, 60],
            azimuth=0.0,
            spp=1,
            srf=InterpolatedSpectrum(
                wavelengths=[550.0, 560.0] * ureg.nm,
                values=[1.0, 1.0] * ureg.dimensionless,
            ),
        ),
    )
    exp.process()
    measure_index = 0
    measure = exp.measures[measure_index]

    # Apply basic post-processing
    steps = [Gather(var="radiance")]
    if eradiate.mode().is_ckd:
        steps.append(
            AggregateCKDQuad(
                var="radiance",
                measure=measure,
                binset=exp.spectral_set[measure_index],
            )
        )
    values = Pipeline(steps=steps).transform(measure.mi_results)

    step = AddSpectralResponseFunction(measure=measure)
    result = step.transform(values)

    # The spectral response function is added to the dataset as a data variable
    assert "srf" in result.data_vars
    # Its only dimension is wavelength
    assert set(result.srf.dims) == {"srf_w"}
