import numpy as np
import pytest

from eradiate.contexts import KernelDictContext
from eradiate.exceptions import UnsupportedModeError
from eradiate.scenes.core import KernelDict
from eradiate.scenes.measure import measure_factory
from eradiate.scenes.measure._distant_array import (
    DistantArrayMeasure,
    DistantArrayReflectanceMeasure,
)


def test_distant_array_radiance(modes_all_mono_ckd):
    # Test default constructor
    d = DistantArrayMeasure()

    ctx = KernelDictContext()
    assert KernelDict.from_elements(d, ctx=ctx).load() is not None

    # Test multiple directions
    d = DistantArrayMeasure(
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], id="testmeasure"
    )
    assert np.allclose(d.directions, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    si = d.sensor_infos()
    assert len(si) == 3

    ctx = KernelDictContext()
    kernel_dict = KernelDict.from_elements(d, ctx=ctx)
    assert kernel_dict.load() is not None
    assert (
        len([key for key in kernel_dict.keys() if key.startswith("testmeasure")]) == 3
    )

    # Test from_angles constructor
    d = DistantArrayMeasure.from_angles([(0, 0), (90, 0)])
    assert np.allclose(d.directions, [[0, 0, 1], [1, 0, 0]])

    ctx = KernelDictContext()
    assert KernelDict.from_elements(d, ctx=ctx).load() is not None

    # Dict construction
    measure_dict = {
        "type": "distant_array",
        "directions": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    d = measure_factory.convert(measure_dict)
    assert np.allclose(d.directions, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    measure_dict = {
        "type": "distant_array",
        "construct": "from_angles",
        "angles": [(0, 0), (90, 0)],
    }
    d = measure_factory.convert(measure_dict)
    assert np.allclose(d.directions, [[0, 0, 1], [1, 0, 0]])


def test_distant_array_reflectance(modes_all_mono_ckd):
    # Test default constructor
    d = DistantArrayReflectanceMeasure()

    ctx = KernelDictContext()
    assert KernelDict.from_elements(d, ctx=ctx).load() is not None

    # Test multiple directions
    d = DistantArrayMeasure(
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], id="testmeasure"
    )
    assert np.allclose(d.directions, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    si = d.sensor_infos()
    assert len(si) == 3

    ctx = KernelDictContext()
    kernel_dict = KernelDict.from_elements(d, ctx=ctx)
    assert kernel_dict.load() is not None
    assert (
        len([key for key in kernel_dict.keys() if key.startswith("testmeasure")]) == 3
    )

    # Test from_angles constructor
    d = DistantArrayReflectanceMeasure.from_angles([(0, 0), (90, 0)])
    assert np.allclose(d.directions, [[0, 0, 1], [1, 0, 0]])

    ctx = KernelDictContext()
    assert KernelDict.from_elements(d, ctx=ctx).load() is not None

    # Dict construction
    measure_dict = {
        "type": "distant_array_reflectance",
        "directions": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    d = measure_factory.convert(measure_dict)
    assert np.allclose(d.directions, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    measure_dict = {
        "type": "distant_array_reflectance",
        "construct": "from_angles",
        "angles": [(0, 0), (90, 0)],
    }
    d = measure_factory.convert(measure_dict)
    assert np.allclose(d.directions, [[0, 0, 1], [1, 0, 0]])


def test_azimuthal_ring_constructor(modes_all_mono_ckd):
    # Default direction, azim resolution
    d = DistantArrayMeasure.azimuthal_ring(zenith_angle=90, azimuth_resolution=90)
    assert np.allclose(d.directions, [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])

    # Default direction, azim steps
    d = DistantArrayMeasure.azimuthal_ring(zenith_angle=45, azimuth_steps=180)
    assert len(d.directions) == 180

    # Angles translate correctly into directions
    d = DistantArrayMeasure.azimuthal_ring(zenith_angle=45, azimuth_steps=360)
    angles_derived = [
        (np.arccos(direction[2]), np.arctan2(direction[1], direction[0]))
        for direction in d.directions
    ]
    assert np.allclose([theta for (theta, _) in angles_derived], np.deg2rad(45))
    # compare cos(phi) to circumvent the issue of phi wrapping around from pi
    # -pi
    assert np.allclose(
        [np.cos(phi) for (_, phi) in angles_derived],
        [np.cos(phi_ref) for phi_ref in np.linspace(0, 2 * np.pi, 360, endpoint=False)],
        atol=0.001,
    )

    # Custom direction
    d = DistantArrayMeasure.azimuthal_ring(
        direction=[1, 0, 0], zenith_angle=90, azimuth_resolution=90
    )
    assert np.allclose(d.directions, [[0, 0, 1], [0, 1, 0], [0, 0, -1], [0, -1, 0]])

    # Dict construction
    measure_dict = {
        "type": "distant_array",
        "id": "test",
        "construct": "azimuthal_ring",
        "zenith_angle": 90,
        "azimuth_resolution": 90,
    }
    d = measure_factory.convert(measure_dict)
    assert np.allclose(d.directions, [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])


def test_distant_unsupported_mode(modes_all_rgb):
    with pytest.raises(UnsupportedModeError):
        d = DistantArrayReflectanceMeasure()
    with pytest.raises(UnsupportedModeError):
        d = DistantArrayMeasure()
