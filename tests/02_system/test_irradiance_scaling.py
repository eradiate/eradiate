from pathlib import Path

import attrs
import dateutil
import numpy as np
import pytest
from numpy.testing import assert_allclose
from skyfield.api import Loader, utc

import eradiate
from eradiate import unit_registry as ureg
from eradiate.config import settings
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
    if datetime is not None:
        # Use Eradiate's Skyfield cache directory
        skyfield_cache_dir = Path(settings["data_path"]) / "cached" / "skyfield"
        skyfield_cache_dir.mkdir(parents=True, exist_ok=True)
        loader = Loader(skyfield_cache_dir)

        # Load JPL ephemeris and timescale
        ts = loader.timescale()
        eph = loader("de421.bsp")

        # Get Earth and Sun positions
        earth = eph["earth"]
        sun = eph["sun"]

        # Convert datetime to skyfield time (ensure UTC timezone)
        dt = dateutil.parser.parse(datetime)
        dt_utc = dt.replace(tzinfo=utc) if dt.tzinfo is None else dt
        t = ts.from_datetime(dt_utc)

        # Calculate Earth-Sun distance in AU
        astrometric = earth.at(t).observe(sun)
        distance_au = astrometric.distance().au

        # Compute scaling factor
        scale_datetime = (1.0 / distance_au) ** 2
    else:
        scale_datetime = 1.0
    expected_radiance = (
        reference_irradiance.m * scale * scale_datetime * reflectance / np.pi
    )
    assert_allclose(result.radiance.values, expected_radiance, rtol=0.01)

    # The BRF is independent of the scaling factor
    assert_allclose(result.brf.values, reflectance)
