import numpy as np

import eradiate.kernel
from ...exceptions import KernelVariantError

_SUPPORTED_VARIANTS = {"scalar_mono", "scalar_mono_double"}


def _check_variant():
    variant = eradiate.kernel.variant()
    if variant not in _SUPPORTED_VARIANTS:
        raise KernelVariantError(f"unsupported kernel variant '{variant}'")


def runner(kernel_dict):
    """Low-level runner function. Takes a kernel dictionary, instantiates the
    corresponding kernel scene and runs the integrator with all sensors.

    .. important::

       Prior to usage, a kernel variant must be selected using
       :func:`~mitsuba.set_variant`.

    Parameter ``kernel_dict`` (:class:.KernelDict`):
        Dictionary describing the kernel scene.

    Returns → dict:
        Dictionary mapping sensor IDs to the corresponding recorded data.
        Sensors without an ID are assigned a default key.
    """
    _check_variant()

    # Basic setup
    from eradiate.kernel.core import LogLevel, Thread

    Thread.thread().logger().set_log_level(LogLevel.Warn)

    results = dict()

    # Run computation
    from eradiate.kernel.core.xml import load_dict

    kernel_scene = load_dict(kernel_dict.data)

    for i_sensor, sensor in enumerate(kernel_scene.sensors()):
        kernel_scene.integrator().render(kernel_scene, sensor)

        # Collect results
        film = sensor.film()
        result = np.array(film.bitmap(), dtype=float)

        sensor_id = str(sensor.id())
        if not sensor_id:
            sensor_id = f"__sensor_{i_sensor}"
        results[sensor_id] = result

    return results
