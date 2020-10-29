"""Runner classes to solve one-dimensional problems."""

import attr
import numpy as np

import eradiate.kernel

from ...scenes.core import KernelDict
from ...util.collections import frozendict
from ...util.exceptions import KernelVariantError


@attr.s
class OneDimRunner:
    r"""Execute simulations on one-dimensional scenes, *i.e.* with 2
    translational invariances.

    .. important::

        Prior to using :class:`OneDimSolver`, a kernel variant must be selected
        using :func:`~mitsuba.set_variant`:

        .. code:: python

            import eradiate.kernel
            from eradiate.solvers.onedim import OneDimSolver

            eradiate.kernel.set_variant("scalar_mono_double")
            solver = OneDimSolver()
            solver.run()

    .. rubric:: Constructor arguments / instance attributes

    ``kernel_dict`` (:class:`~eradiate.scenes.core.KernelDict`):
        Dictionary used to generate the scene for which simulations will be
        run. If no value is passed, a default scene is constructed. It
        consists of a square covering :math:`[-1, 1]^2` with normal vector
        :math:`+Z` and a Lambertian BRDF (reflectance :math:`\rho = 0.5`)
        illuminated by a directional emitter with direction vector
        :math:`-Z` and constant irradiance equal to 1. By default the
        sensor will record four points, coressponding to a zenith angle of
        :math:`45°` and azimuth angles of :math:`[0°, 90°, 180°, 270°]`.
        If set to `None`, defaults to :data:`DEFAULT_KERNEL_DICT`.
    """

    SUPPORTED_VARIANTS = frozenset({"scalar_mono", "scalar_mono_double"})
    DEFAULT_KERNEL_DICT = frozendict({
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
        "measure": {
            "type": "radiancemeterarray",
            "origins": "0, 0, 0.1, 0, 0, 0.1, 0, 0, 0.1, 0, 0, 0.1",
            "directions": "1, 0, -1, -1, 0, -1, 0, 1, -1, 0, -1, -1",
            "id": "measure",
            "sampler": {
                "type": "independent",
                "sample_count": 32
            },
            "film": {
                "type": "hdrfilm",
                "width": 4,
                "height": 1,
                "pixel_format": "luminance",
                "component_format": "float32",
                "rfilter": {"type": "box"}
            }
        },
        "integrator": {"type": "path"}
    })

    kernel_dict = attr.ib(default=None)

    def _check_variant(self):
        variant = eradiate.kernel.variant()
        if variant not in self.SUPPORTED_VARIANTS:
            raise KernelVariantError(f"unsupported kernel variant '{variant}'")

    def __attrs_post_init__(self):
        if self.kernel_dict is None:
            self.kernel_dict = KernelDict(self.DEFAULT_KERNEL_DICT)

        self.init()

    def init(self):
        """(Re)initialise internal state. Currently a placeholder."""
        pass

    def run(self, show_progress=True):
        """Run the simulation for a set of specified sensor angular
        configurations.

        The solver uses the variant stored in its :data:`variant` instance
        attribute.

        Parameter ``show_progress`` (bool):
            If `True`, display a progress bar while running the simulation.
            This option is currently not available.

        Returns → dict:
            Maps the sensor's ids to their recorded leaving radiance.
        """
        self._check_variant()

        # Basic setup
        from eradiate.kernel.core import LogLevel, Thread
        Thread.thread().logger().set_log_level(LogLevel.Warn)

        results = dict()

        # Run computation
        kernel_scene = self.kernel_dict.load()
        for sensor in kernel_scene.sensors():
            kernel_scene.integrator().render(kernel_scene, sensor)

            # Collect results
            film = sensor.film()
            result = np.array(film.bitmap(), dtype=float)

            results[f"{sensor.id()}"] = result

        return results
