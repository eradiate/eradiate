import astropy
import attrs
import dateutil
import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.exceptions import UnsupportedModeError
from eradiate.scenes.measure import MeasureSpectralConfig


@pytest.mark.slow
@pytest.mark.parametrize(
    "measure",
    [
        {"type": "hdistant"},
        {
            "type": "mdistant",
            "construct": "from_viewing_angles",
            "zeniths": np.arange(-75, 76, 5),
            "azimuths": 0.0,
        },
    ],
    ids=["hdistant", "mdistant"],
)
@pytest.mark.parametrize("scale", [1.0, 0.5])
@pytest.mark.parametrize("datetime", [None, "2000-01-01"])
def test_radiance_scaling(modes_all_double, measure, scale, datetime):
    """
    Check if the recorded radiance scales as expected with the incoming
    irradiance, while the reflectance remains constant.
    """

    reflectance = 0.5
    irradiance_spectrum = eradiate.scenes.spectra.SolarIrradianceSpectrum(
        dataset="thuillier_2003"
    )

    if eradiate.mode().is_mono:
        spectral_cfg = MeasureSpectralConfig.new(wavelengths=[550.0] * ureg.nm)
    elif eradiate.mode().is_ckd:
        spectral_cfg = MeasureSpectralConfig.new(bin_set="1nm", bins=["550"])
    else:
        raise UnsupportedModeError

    reference_irradiance = irradiance_spectrum.eval(
        spectral_ctx=spectral_cfg.spectral_ctxs()[0]
    )

    measure["spectral_cfg"] = spectral_cfg
    exp = eradiate.experiments.AtmosphereExperiment(
        surface={"type": "lambertian", "reflectance": reflectance},
        atmosphere=None,
        illumination={
            "type": "directional",
            "irradiance": attrs.evolve(
                irradiance_spectrum, scale=scale, datetime=datetime
            ),
        },
        measures=measure,
    )
    result = eradiate.run(exp)

    # The radiance is proportional to the scaling factor
    scale_datetime = (
        (
            float(
                astropy.coordinates.get_sun(
                    astropy.time.Time(dateutil.parser.parse(datetime))
                ).distance
                / astropy.units.au
            )
            ** 2
        )
        if datetime is not None
        else 1.0
    )
    expected_radiance = (
        reference_irradiance.m * scale * scale_datetime * reflectance / np.pi
    )
    assert np.allclose(result.radiance.values, expected_radiance)

    # The BRF is independent of the scaling factor
    assert np.allclose(result.brf.values, reflectance)
