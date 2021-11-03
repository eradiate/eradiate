import pytest

from eradiate.exceptions import KernelVariantError
from eradiate.experiments import mitsuba_run
from eradiate.scenes.core import KernelDict


def test_mitsuba_run(modes_all):
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
    results = mitsuba_run(kernel_dict)

    # We store film values and SPPs
    assert "values" in results
    assert "spp" in results

    # Sensors without ID have a default key
    assert "__sensor_0" in results["values"]
    assert "__sensor_0" in results["spp"]
    # Sensors with ID have their ID as key
    assert "my_sensor" in results["values"]
    assert "my_sensor" in results["spp"]


def test_mitsuba_run_fail():
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
        mitsuba_run(kernel_dict)
