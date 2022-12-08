import mitsuba as mi
import pytest

from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import traverse
from eradiate.scenes.phase import HenyeyGreensteinPhaseFunction, PhaseFunction
from eradiate.test_tools.types import check_type


def test_hg_type():
    check_type(
        HenyeyGreensteinPhaseFunction,
        expected_mro=[HenyeyGreensteinPhaseFunction, PhaseFunction],
        expected_slots=[],
    )


@pytest.mark.parametrize("kwargs", [{}, {"g": 0.25}], ids=["noargs", "args"])
def test_hg(modes_all_double, kwargs):
    phase = HenyeyGreensteinPhaseFunction(**kwargs)

    # Produced kernel dictionary is valid
    template, params = traverse(phase)
    kernel_dict = template.render(ctx=KernelDictContext())
    assert isinstance(mi.load_dict(kernel_dict), mi.PhaseFunction)
