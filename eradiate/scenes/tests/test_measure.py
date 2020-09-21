import numpy as np

from eradiate.scenes.core import KernelDict
from eradiate.scenes.measure import DistantMeasure, PerspectiveCameraMeasure
from eradiate.util.units import ureg


def test_distant_class(mode_mono):
    # Constructor
    d = DistantMeasure()
    assert KernelDict.empty().add(d).load() is not None


def test_perspective_class(mode_mono):
    # Constructor
    d = PerspectiveCameraMeasure()
    assert KernelDict.empty().add(d) is not None
