import attr
import eradiate
import numpy as np

from ..scenes.atmosphere import RayleighHomogeneous
from ..scenes.builder import *
from ..scenes.util import angles_to_direction
from ..util import ensure_array, ureg


@ureg.wraps(None, (ureg.deg, ureg.deg, None), strict=False)
def _make_distant(zenith=0., azimuth=0., spp=10000):
    """Create a `distant` plugin interface instance.

    :param (float) zenith: Zenith angle [deg].
    :param (float) zenith: Azimuth angle angle [deg].
    :param (int) spp: Number of samples used from this sensor.

    :return (Distant): A Distant sensor plugin interface facing the direction
        specified by the angular configuration and pointing towards the origin
        :math:`(0, 0, 0)` in world coordinates.
    """

    film = films.HDRFilm(
        width=1,
        height=1,
        pixel_format="luminance",
        rfilter=rfilters.Box()
    )

    sensor = sensors.Distant(
        direction=-angles_to_direction(theta=np.deg2rad(zenith),
                                       phi=np.deg2rad(azimuth)),
        target=[0, 0, 0],
        sampler=samplers.Independent(sample_count=spp),
        film=film
    )

    return sensor


def _make_default_scene():
    bsdf = bsdfs.Diffuse(id="brdf_surface", reflectance=Spectrum(0.5))
    emitter = emitters.Directional(
        direction=[0, 0, -1], irradiance=Spectrum(1.0))

    scene = Scene(
        bsdfs=[bsdf],
        shapes=[shapes.Rectangle(bsdf=Ref(id="brdf_surface"))],
        emitter=emitter,
        integrator=integrators.Path()
    )

    return scene


@attr.s
class OneDimSolver:
    r"""This class executes simulations on one-dimensional scenes, *i.e.* with 
    2 translational invariances.

    Constructor arguments / public attributes:
        ``scene`` (:class:`eradiate.scenes.builder.Scene`):
            Scene for which simulations will be run. If no value is 
            passed, a default scene is constructed. It consists of a square 
            covering :math:`[-1, 1]^2` with normal vector :math:`+Z` 
            with a Lambertian BRDF (reflectance :math:`\rho = 0.5`) illuminated
            directional emitter with direction vector :math:`-Z` and constant 
            irradiance equal to 1.
    """

    scene = attr.ib(default=_make_default_scene())

    def run(self, vza=0., vaa=0., spp=3200):
        """Run the simulation for a set of specified sensor angular
        configurations.

        Parameter ``vza`` (float or array-like):
            Viewing zenith angles [deg].
        
        Parameter ``vaa`` (float or array-like): 
            Viewing azimuth angles [deg].

        Parameter ``spp`` (int): 
            Number of samples taken for each angular configuration.

        Returns → float or array: 
            Recorded leaving radiance.
        """

        # Ensure that vza and vaa are numpy arrays
        vza = ensure_array(vza)
        vaa = ensure_array(vaa)

        # Basic setup
        eradiate.kernel.set_variant("scalar_mono")
        from eradiate.kernel.core import Thread
        Thread.thread().logger().clear_appenders()

        reflected_radiance = np.empty((len(vza), len(vaa)))

        for i, theta in enumerate(vza):
            for j, phi in enumerate(vaa):
                # Adjust scene setup
                scene_xml = Scene.convert(self.scene)
                scene_xml.sensor = _make_distant(theta, phi, spp)

                # Run computation
                scene = scene_xml.instantiate()
                sensor = scene.sensors()[0]
                scene.integrator().render(scene, sensor)

                # Collect results
                film = sensor.film()
                result = float(np.array(film.bitmap(), dtype=float))
                reflected_radiance[i, j] = result

        # Fix result dimensionality (remove useless dims)
        try:
            return float(reflected_radiance)
        except TypeError:
            return np.squeeze(reflected_radiance)


@attr.s
class RayleighHomogeneousSolver(OneDimSolver):
    """
    Solver for Rayleigh scattering homogeneous one-dimensional atmospheres
    """

    def __attrs_post_init__(self):
        self.init()

    def init(self):
        """Initialise internal state."""

        width = 2
        height = 2
        atmosphere = RayleighHomogeneous()

        surface_bsdfs = [
            bsdfs.Diffuse(id="surface_brdf", reflectance=Spectrum(0.5))
        ]
        surface_shapes = [
            shapes.Rectangle(bsdf=Ref(id="surface_brdf"),
                             to_world=Transform([Scale(width / 2)]))
        ]

        emitter = emitters.Directional(
            direction=[0, 0, -1],
            irradiance=Spectrum(1.0)
        )

        scene = Scene(
            bsdfs=surface_bsdfs,
            phase=atmosphere.phase(),
            media=atmosphere.media(),
            shapes=surface_shapes + atmosphere.shapes(),
            emitter=emitter,
            integrator=integrators.VolPath(),
        )

        self.scene = scene
