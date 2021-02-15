"""Runner classes to solve one-dimensional problems."""

import attr
import numpy as np

import eradiate.kernel
from ...scenes.core import KernelDict
from ...util.attrs import documented, parse_docs
from ...util.exceptions import KernelVariantError


@parse_docs
@attr.s
class OneDimRunner:
    r"""
    Execute simulations on one-dimensional scenes, *i.e.* with 2 translational
    invariances.

    .. important::

        Prior to using :class:`OneDimSolver`, a kernel variant must be selected
        using :func:`~mitsuba.set_variant`:

        .. code:: python

            import eradiate.kernel
            from eradiate.solvers.onedim import OneDimSolver

            eradiate.kernel.set_variant("scalar_mono_double")
            solver = OneDimSolver()
            solver.run()
    """

    SUPPORTED_VARIANTS = frozenset({"scalar_mono", "scalar_mono_double"})

    kernel_dict = documented(
        attr.ib(
            factory=KernelDict.empty,
            converter=KernelDict,
        ),
        doc="Dictionary used to generate the scene for which simulations will be "
            "run. If no value is passed, a default scene is constructed. It "
            "consists of a square covering :math:`[-1, 1]^2` with normal vector "
            ":math:`+Z` and a Lambertian BRDF (reflectance :math:`\\rho = 0.5`) "
            "illuminated by a directional emitter with direction vector "
            ":math:`-Z` and constant irradiance equal to 1. By default the "
            "sensor will record four points, coressponding to a zenith angle of "
            ":math:`45°` and azimuth angles of :math:`[0°, 90°, 180°, 270°]`. "
            "If set to `None`, defaults to :data:`DEFAULT_KERNEL_DICT`.",
        type=":class:`~eradiate.scenes.core.KernelDict`"
    )

    def _check_variant(self):
        variant = eradiate.kernel.variant()
        if variant not in self.SUPPORTED_VARIANTS:
            raise KernelVariantError(f"unsupported kernel variant '{variant}'")

    def run(self):
        """Run the simulation for a set of specified sensor angular
        configurations.

        The solver uses the variant stored in its :data:`variant` instance
        attribute.

        Returns → dict:
            Maps the sensor's IDs to their recorded leaving radiance.
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
