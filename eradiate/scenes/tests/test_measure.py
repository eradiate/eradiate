import numpy as np

from eradiate.scenes.core import KernelDict
from eradiate.scenes.measure import DistantMeasure, PerspectiveCameraMeasure, RadianceMeterHemisphere
from eradiate.util.units import ureg


def test_distant(mode_mono):
    # Constructor
    d = DistantMeasure()
    assert KernelDict.empty().add(d).load() is not None


def test_perspective(mode_mono):
    # Constructor
    d = PerspectiveCameraMeasure()
    assert KernelDict.empty().add(d).load() is not None


def test_radiancemeter_hemispherical(mode_mono):
    # Test constructor
    d = RadianceMeterHemisphere()
    assert KernelDict.empty().add(d).load() is not None

    # Test repack method
    data = np.linspace(0, 1, 9*36)

    d = RadianceMeterHemisphere(zenith_res=10, azimuth_res=10)
    data_reshaped = d.repack_results(data)

    assert np.shape(data_reshaped) == (9, 36)

    # Test hemisphere selection
    d = RadianceMeterHemisphere()
    d_back = RadianceMeterHemisphere(hemisphere="back")

    directions_front = d.directions()
    directions_back = d_back.directions()

    assert np.allclose(-directions_front, directions_back)
