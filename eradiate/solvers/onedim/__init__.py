""" Basic facilities to run simulations on one-dimensional scenes. """

import attr
import numpy as np

import eradiate.kernel
from ...scenes import measure, SceneDict
from ...util import ensure_array
from ...util.collections import frozendict
from ...util.exceptions import KernelVariantError


@attr.s
class OneDimSolver:
    r"""Execute simulations on one-dimensional scenes, *i.e.* with 2
    translational invariances.

    .. important::

        Prior to calling :meth:`run`, a kernel variant must be selected using
        :func:`~mitsuba.set_variant`:

        .. code:: python

            import eradiate.kernel
            from eradiate.solvers.onedim import OneDimSolver

            eradiate.kernel.set_variant("scalar_mono_double")
            solver = OneDimSolver()
            solver.run()

    Constructor arguments / public attributes:
        ``scene_dict`` (:class:`~eradiate.scenes.SceneDict`):
            Dictionary used to generate the scene for which simulations will be 
            run. If no value is passed, a default scene is constructed. It 
            consists of a square covering :math:`[-1, 1]^2` with normal vector 
            :math:`+Z` and a Lambertian BRDF (reflectance :math:`\rho = 0.5`) 
            illuminated by a directional emitter with direction vector 
            :math:`-Z` and constant irradiance equal to 1. If set to `None`,
            defaults to :data:`DEFAULT_DICT_SCENE`.
    """

    SUPPORTED_VARIANTS = frozenset({"scalar_mono", "scalar_mono_double"})
    DEFAULT_SCENE_DICT = frozendict({
        "type": "scene",
        "bsdf_surface": {
            "type": "diffuse",
            "reflectance": {"type": "uniform", "value": 0.5}
        },
        "surface": {
            "type": "rectangle",
            "bsdf": {"type": "ref", "id": "bsdf_surface"}
        },
        "illumination": {
            "type": "directional",
            "direction": [0, 0, -1],
            "irradiance": {"type": "uniform", "value": 1.0}
        },
        "integrator": {"type": "path"}
    })

    scene_dict = attr.ib(default=None)

    def _check_variant(self):
        variant = eradiate.kernel.variant()
        if variant not in self.SUPPORTED_VARIANTS:
            raise KernelVariantError(f"unsupported kernel variant '{variant}'")

    def __attrs_post_init__(self):
        if self.scene_dict is None:
            self.scene_dict = SceneDict(self.DEFAULT_SCENE_DICT)

        self.init()

    def init(self):
        """(Re)initialise internal state. Currently a placeholder."""
        pass

    def run(self, vza=0., vaa=0., spp=3200, squeeze=True):
        """Run the simulation for a set of specified sensor angular
        configurations.

        The solver uses the variant stored in its :data:`variant` instance
        attribute.

        Parameter ``vza`` (float or array-like):
            Viewing zenith angles [deg].

        Parameter ``vaa`` (float or array-like):
            Viewing azimuth angles [deg].

        Parameter ``spp`` (int):
            Number of samples taken for each angular configuration.

        Returns → float or array:
            Recorded leaving radiance.
        """
        self._check_variant()

        # Ensure that vza and vaa are numpy arrays
        vza = ensure_array(vza)
        vaa = ensure_array(vaa)

        # Basic setup
        from eradiate.kernel.core import Thread
        Thread.thread().logger().clear_appenders()

        reflected_radiance = np.empty((len(vza), len(vaa)))

        for i, theta in enumerate(vza):
            for j, phi in enumerate(vaa):
                # Adjust scene setup
                self.scene_dict.add(measure.Distant(
                    {'zenith': theta, 'azimuth': phi, 'spp': spp}
                ))

                # Run computation
                kernel_scene = self.scene_dict.load()
                sensor = kernel_scene.sensors()[0]
                kernel_scene.integrator().render(kernel_scene, sensor)

                # Collect results
                film = sensor.film()
                result = float(np.array(film.bitmap(), dtype=float))
                reflected_radiance[i, j] = result

        if squeeze:
            # Fix result dimensionality (remove useless dims)
            try:
                return float(reflected_radiance)
            except TypeError:
                return np.squeeze(reflected_radiance)
        else:
            return reflected_radiance