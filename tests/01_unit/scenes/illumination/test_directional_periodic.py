import mitsuba as mi
import pytest

from eradiate.scenes.illumination import DirectionalPeriodicIllumination
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "tested",
    [
        {},
        {
            "zenith": 1.0,
            "azimuth": 2.0,
            "periodic_box": {"min": [-1, -1, -1], "max": [1, 1, 1]},
        },
    ],
    ids=[
        "no_args",
        "args",
    ],
)
def test_directional_kernel_dict(modes_all_double, tested):
    # The associated kernel dict is correctly formed and can be loaded
    illumination = DirectionalPeriodicIllumination(**tested)
    check_scene_element(illumination, mi_cls=mi.Emitter)
