from typing import Dict, List, Optional

import mitsuba
import numpy as np

from ...exceptions import KernelVariantError
from ...scenes.core import KernelDict

_SUPPORTED_VARIANTS = {"scalar_mono", "scalar_mono_double"}


def _check_variant():
    variant = mitsuba.variant()
    if variant not in _SUPPORTED_VARIANTS:
        raise KernelVariantError(f"unsupported kernel variant '{variant}'")


def runner(kernel_dict: KernelDict, sensor_ids: Optional[List[str]] = None) -> Dict:
    """
    Low-level runner function. Takes a kernel dictionary, instantiates the
    corresponding kernel scene and runs the integrator with all sensors.

    .. important:: Prior to usage, a kernel variant must be selected using
       :func:`~mitsuba.set_variant`.

    Parameter ``kernel_dict`` (:class:`.KernelDict`):
        Dictionary describing the kernel scene.

    Parameter ``sensor_ids`` (list[str] or None):
        Identifiers of the sensors for which the integrator is run. If set to
        ``None``, all sensors are selected.

    Returns → dict:
        Nested dictionaries with the following structure:

        .. code:: python

           {
               "values": {
                   "sensor_0": data_0, # Arrays of shape (width, height, n_channels)
                   "sensor_1": data_1, # (n_channels == 1 for mono variants, possibly more)
                   ...
               },
               "spp": {
                   "sensor_0": sample_count_0,
                   "sensor_1": sample_count_1,
                   ...
               },
           }
    """
    _check_variant()

    # Result storage
    results = dict()

    # Run computation
    from mitsuba.core.xml import load_dict

    kernel_scene = load_dict(kernel_dict.data)

    # Define the list of processed sensors
    if sensor_ids is None:
        sensors = kernel_scene.sensors()
    else:
        sensors = [
            sensor
            for sensor in kernel_scene.sensors()
            if str(sensor.id()) in sensor_ids
        ]

    # Run kernel for selected sensors
    for i_sensor, sensor in enumerate(sensors):
        # Run Mitsuba
        kernel_scene.integrator().render(kernel_scene, sensor)

        # Collect results
        film = sensor.film()
        result = np.array(film.bitmap(), dtype=float)

        sensor_id = str(sensor.id())
        if not sensor_id:  # Assign default ID if sensor doesn't have one
            sensor_id = f"__sensor_{i_sensor}"

        if "values" not in results:
            results["values"] = {}
        results["values"][sensor_id] = result

        # Add sensor SPPs
        if "spp" not in results:
            results["spp"] = {}
        results["spp"][sensor_id] = sensor.sampler().sample_count()

    return results
