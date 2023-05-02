import mitsuba as mi
import pytest

from eradiate import KernelContext
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

    kdict = template.render(ctx=KernelContext())
    assert "medium" not in kdict

    kdict = template.render(
        ctx=KernelContext(kwargs={"measure.atmosphere_medium_id": "test_atmosphere"})
    )
    assert kdict["medium"] == {"type": "ref", "id": "test_atmosphere"}
