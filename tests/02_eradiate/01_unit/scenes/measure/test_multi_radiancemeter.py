import mitsuba as mi
import pytest

from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import traverse
from eradiate.scenes.measure import MultiRadiancemeterMeasure
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "tested",
    [{}, dict(origins=[[0, 0, 0]] * 3, directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])],
    ids=[
        "no_args",
        "origins_directions",
    ],
)
def test_multi_radiancemeter(modes_all_double, tested):
    measure = MultiRadiancemeterMeasure(**tested)
    check_scene_element(measure, mi.Sensor)


def test_multi_radiancemeter_medium(mode_mono):
    measure = MultiRadiancemeterMeasure()
    template, _ = traverse(measure)

    ctx1 = KernelDictContext()
    ctx2 = KernelDictContext(kwargs={"measure.atmosphere_medium_id": "test_atmosphere"})

    kd1 = template.render(ctx=ctx1, drop=True)
    assert "medium" not in kd1

    kd2 = template.render(ctx=ctx2, drop=True)
    assert kd2["medium"] == {"type": "ref", "id": "test_atmosphere"}
