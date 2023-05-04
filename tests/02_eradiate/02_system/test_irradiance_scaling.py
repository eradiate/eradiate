import astropy
import attrs
import dateutil
import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.spectral.index import SpectralIndex


@pytest.mark.slow
@pytest.mark.parametrize(
    "measure",
    [
        {"type": "hdistant"},
        {
            "type": "mdistant",
            "construct": "hplane",
            "zeniths": np.arange(-75, 76, 5),
            "azimuth": 0.0,
            "srf": {
                "type": "multi_delta",
                "wavelengths": 550.0 * ureg.nm,
            },
        },
    ],
    ids=["hdistant", "mdistant"],
)
@pytest.mark.parametrize("scale", [1.0, 0.5])
@pytest.mark.parametrize("datetime", [None, "2000-01-01"])
def test_radiance_scaling(modes_all_double, measure, scale, datetime):
    """
    Radiance scaling
    ================

    This test checks if the recorded radiance scales as expected with the
    incoming irradiance, while the reflectance remains constant.

    Rationale
    ---------

    An experiment is set up in which the illumination uses an irradiance
    spectrum to which a scaling is applied.

    Expected behaviour
    ------------------

    The scaling is applied to the illumination; consequently, the recorded
    radiance is scaled by the same amount.
    """

    reflectance = 0.5
    irradiance_spectrum = eradiate.scenes.spectra.SolarIrradianceSpectrum(
        dataset="thuillier_2003"
    )

    reference_irradiance = irradiance_spectrum.eval(SpectralIndex.new())

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
                astropy.units.au
                / astropy.coordinates.get_sun(
                    astropy.time.Time(dateutil.parser.parse(datetime))
                ).distance
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
