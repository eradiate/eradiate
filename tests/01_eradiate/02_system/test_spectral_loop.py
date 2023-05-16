import numpy as np
import pytest

import eradiate
from eradiate.experiments import AtmosphereExperiment, CanopyExperiment
from eradiate.scenes.bsdfs import LambertianBSDF
from eradiate.scenes.illumination import DirectionalIllumination
from eradiate.scenes.measure import MultiDistantMeasure
from eradiate.scenes.spectra import SolarIrradianceSpectrum
from eradiate.units import unit_registry as ureg


@pytest.mark.parametrize("cls", ["onedim", "rami"])
@pytest.mark.parametrize(
    "wavelengths",
    [
        [500.0] * ureg.nm,
        [500.0 + 100 * i for i in range(11)] * ureg.nm,
        [500.0 + 100.0 * i for i in range(10, -1, -1)] * ureg.nm,
    ],
    ids=["single", "multiple_ascending", "multiple_descending"],
)
@pytest.mark.parametrize("irradiance", ["uniform", "solar"])
def test_spectral_loop(mode_mono, cls, wavelengths, irradiance):
    """
    Spectral loop
    =============

    This test case checks if the spectral loop implementation and
    post-processing behaves as intended.

    Rationale
    ---------

    The scene consists of a square Lambertian surface with reflectance
    :math:`\\rho = 1` illuminated by a directional emitter positioned at the
    zenith. The reflected radiance is computed in monochromatic mode for a
    single wavelength and for multiple wavelength. The experiment is repeated:

    * for all solver applications supporting this type of scene;
    * for several spectral configurations (single wavenlength, multiple
      ascending, multiple descending);
    * for several irradiance spectra (uniform, solar).

    Expected behaviour
    ------------------

    All BRF values are equal to 1.
    """
    if cls == "onedim":
        cls_exp = AtmosphereExperiment
        kwargs = {"atmosphere": None}
    elif cls == "rami":
        cls_exp = CanopyExperiment
        kwargs = {"canopy": None}
    else:
        raise ValueError

    if irradiance == "uniform":
        illumination = DirectionalIllumination(irradiance=1.0)
    elif irradiance == "solar":
        illumination = DirectionalIllumination(irradiance=SolarIrradianceSpectrum())

    exp = cls_exp(
        surface=LambertianBSDF(reflectance=1.0),
        illumination=illumination,
        measures=[
            MultiDistantMeasure(
                spp=1,
                srf={"type": "multi_delta", "wavelengths": wavelengths},
            )
        ],
        **kwargs,
    )

    results = eradiate.run(exp)
    assert np.allclose(np.squeeze(results.brf.values), 1.0)
