import pytest

from eradiate.exceptions import KernelVariantError
from eradiate.scenes.core import KernelDict
from eradiate.solvers.core import runner


def test_runner(mode_mono):
    # Use on a simple scene
    kernel_dict = KernelDict(
        {
            "type": "scene",
            "shape": {"type": "rectangle"},
            "illumination": {"type": "constant"},
            "sensor_0": {
                "type": "distant",
                "film": {
                    "type": "hdrfilm",
                    "pixel_format": "luminance",
                    "rfilter": {"type": "box"},
                },
            },
            "sensor_1": {
                "id": "my_sensor",
                "type": "distant",
                "film": {
                    "type": "hdrfilm",
                    "pixel_format": "luminance",
                    "rfilter": {"type": "box"},
                },
            },
            "integrator": {"type": "path"},
        }
    )
    results = runner(kernel_dict)

    # Sensors without ID have a default key
    assert "__sensor_0" in results
    # Sensors with ID have their ID as key
    assert "my_sensor" in results


def test_runner_fail():
    kernel_dict = {
        "type": "scene",
        "shape": {"type": "rectangle"},
        "illumination": {"type": "constant"},
        "sensor_0": {"type": "distant"},
        "sensor_1": {"type": "distant", "id": "my_sensor"},
        "integrator": {"type": "path"},
    }

    # Runner raises if unsupported variant is active
    import mitsuba

    mitsuba.set_variant("scalar_rgb")
    with pytest.raises(KernelVariantError):
        runner(kernel_dict)
