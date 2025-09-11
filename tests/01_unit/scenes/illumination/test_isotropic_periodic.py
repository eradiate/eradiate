import mitsuba as mi
import pytest

from eradiate.scenes.illumination import IsotropicPeriodicIllumination
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "tested",
    [
        {},
        {
            "periodic_box": {"min": [-1, -1, -1], "max": [1, 1, 1]},
        },
    ],
    ids=[
        "no_args",
        "args",
    ],
)
def test_isotropic_kernel_dict(modes_all_double, tested):
    # The associated kernel dict is correctly formed and can be loaded
    illumination = IsotropicPeriodicIllumination(**tested)
    check_scene_element(illumination, mi_cls=mi.Emitter)
