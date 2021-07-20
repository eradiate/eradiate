import numpy as np
import pytest

from eradiate.scenes.illumination import DirectionalIllumination
from eradiate.scenes.measure import DistantReflectanceMeasure
from eradiate.scenes.spectra import SolarIrradianceSpectrum
from eradiate.scenes.surface import LambertianSurface
from eradiate.solvers.onedim import OneDimScene, OneDimSolverApp
from eradiate.solvers.rami import RamiScene, RamiSolverApp


@pytest.mark.parametrize("cls", ["onedim", "rami"])
@pytest.mark.parametrize(
    "wavelengths",
    [
        [500.0],
        [500.0 + 100 * i for i in range(11)],
        [500.0 + 100.0 * i for i in range(10, -1, -1)],
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
        cls_app = OneDimSolverApp
        cls_scene = OneDimScene
        kwargs = {"atmosphere": None}
    elif cls == "rami":
        cls_app = RamiSolverApp
        cls_scene = RamiScene
        kwargs = {"canopy": None}
    else:
        raise ValueError

    if irradiance == "uniform":
        illumination = DirectionalIllumination(irradiance=1.0)
    elif irradiance == "solar":
        illumination = DirectionalIllumination(irradiance=SolarIrradianceSpectrum())

    app = cls_app(
        scene=cls_scene(
            surface=LambertianSurface(reflectance=1.0),
            illumination=illumination,
            measures=[
                DistantReflectanceMeasure(
                    film_resolution=(1, 1),
                    spp=1,
                    spectral_cfg={"wavelengths": wavelengths},
                )
            ],
            **kwargs,
        )
    )

    app.run()
    assert np.allclose(np.squeeze(app.results["measure"].brf.values), 1.0)
