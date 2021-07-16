import numpy as np
import pytest

from eradiate.scenes.illumination import DirectionalIllumination
from eradiate.scenes.measure import DistantRadianceMeasure
from eradiate.scenes.surface import LambertianSurface
from eradiate.solvers.onedim import OneDimScene, OneDimSolverApp
from eradiate.solvers.rami import RamiScene, RamiSolverApp


@pytest.mark.parametrize("cls", ["onedim", "rami"])
def test_spectral_loop(mode_mono, cls):
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
    single wavelength and for multiple wavelength. The experiment is repeated
    for all solver applications supporting this type of scene.

    Expected behaviour
    ------------------

    All radiance values are expected to be equal to :math:`1 / \\pi`.
    """
    if cls == "onedim":
        apps = [
            OneDimSolverApp(
                scene=OneDimScene(
                    atmosphere=None,
                    surface=LambertianSurface(reflectance=1.0),
                    illumination=DirectionalIllumination(irradiance=1.0),
                    measures=[
                        DistantRadianceMeasure(
                            spp=1, spectral_cfg={"wavelengths": wavelengths}
                        )
                    ],
                )
            )
            for wavelengths in [[500.0], [100.0 * x for x in range(4, 10)]]
        ]
    elif cls == "rami":
        apps = [
            RamiSolverApp(
                scene=RamiScene(
                    canopy=None,
                    surface=LambertianSurface(reflectance=1.0),
                    illumination=DirectionalIllumination(irradiance=1.0),
                    measures=[
                        DistantRadianceMeasure(
                            spp=1, spectral_cfg={"wavelengths": wavelengths}
                        )
                    ],
                )
            )
            for wavelengths in [[500.0], [100.0 * x for x in range(4, 10)]]
        ]
    else:
        raise ValueError

    for app in apps:
        app.run()

    results = [np.squeeze(app.results["measure"].lo.values) for app in apps]
    assert all([np.allclose(result, 1.0 / np.pi) for result in results])
