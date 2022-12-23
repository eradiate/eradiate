import mitsuba as mi

from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import traverse
from eradiate.scenes.measure import MultiRadiancemeterMeasure
from eradiate.scenes.measure._radiancemeter import RadiancemeterMeasure
from eradiate.test_tools.types import check_scene_element


def test_radiancemeter_construct(mode_mono):
    measure = RadiancemeterMeasure()
    check_scene_element(measure, mi.Sensor)


def test_radiancemeter_medium(mode_mono):
    measure = MultiRadiancemeterMeasure()
    template, _ = traverse(measure)

    ctx1 = KernelDictContext()
    ctx2 = KernelDictContext(kwargs={"atmosphere_medium_id": "test_atmosphere"})

    kd1 = template.render(ctx=ctx1, drop=True)
    assert "medium" not in kd1

    kd2 = template.render(ctx=ctx2, drop=True)
    assert kd2["medium"] == {"type": "ref", "id": "test_atmosphere"}
