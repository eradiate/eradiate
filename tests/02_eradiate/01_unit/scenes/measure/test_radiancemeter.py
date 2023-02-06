import mitsuba as mi

from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import traverse
from eradiate.scenes.measure import RadiancemeterMeasure
from eradiate.test_tools.types import check_scene_element


def test_radiancemeter_construct(mode_mono):
    measure = RadiancemeterMeasure()
    check_scene_element(measure, mi.Sensor)


def test_radiancemeter_medium(mode_mono):
    measure = RadiancemeterMeasure()
    template, _ = traverse(measure)

    ctx1 = KernelDictContext()
    ctx2 = KernelDictContext(kwargs={"measure.atmosphere_medium_id": "test_atmosphere"})

    kdict = template.render(ctx=KernelDictContext())
    assert "medium" not in kdict

    kdict = template.render(
        ctx=KernelDictContext(
            kwargs={"measure.atmosphere_medium_id": "test_atmosphere"}
        )
    )
    assert kdict["medium"] == {"type": "ref", "id": "test_atmosphere"}
