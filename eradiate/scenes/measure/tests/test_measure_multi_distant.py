import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.measure._multi_distant import MultiDistantMeasure


def test_multi_distant_measure_construct(mode_mono):
    """
    Basic constructor testing for MultiDistantMeasure.
    """

    ctx = KernelDictContext()

    # Constructing without argument succeeds
    measure = MultiDistantMeasure()

    # The produced kernel dictionary can be instantiated
    kernel_dict = measure.kernel_dict(ctx)
    assert kernel_dict.load().expand()[0]


def test_multi_distant_measure_viewing_angles(mode_mono):
    """
    Unit tests for :attr:`.MultiDistantMeasure.viewing_angles`.
    """
    # Viewing angle computation is correct
    measure = MultiDistantMeasure(
        directions=[
            [0, 0, -1],
            [-1, 0, -1],
            [0, -1, -1],
            [-1, -1, -1],
        ]
    )

    expected = (
        np.array(
            [
                [0, 0],
                [45, 0],
                [45, 90],
                [np.rad2deg(np.arccos(1 / np.sqrt(3))), 45],
            ]
        ).reshape((4, 1, 2))
        * ureg.deg
    )

    assert np.allclose(expected, measure.viewing_angles)

    # Directions which would normally map to the [-π, 0] domain are normalised
    # to [0, 2π]
    measure = MultiDistantMeasure(directions=[[0, 1, -1]])
    assert np.allclose([45, 270] * ureg.deg, measure.viewing_angles)


def test_multi_distant_measure_from_viewing_angles(mode_mono):
    """
    Unit tests for :meth:`.MultiDistantMeasure.from_viewing_angles`.
    """
    # Construct from viewing angles not in a hemisphere plane cut
    zeniths = [0, 45, 90, 45, 45, 45, 90, 90, 90]
    azimuths = [0, 0, 0, 0, 45, 90, 0, 45, 90]
    angles = np.reshape(np.stack((zeniths, azimuths), axis=-1), (-1, 1, 2)) * ureg.deg

    measure = MultiDistantMeasure.from_viewing_angles(zeniths, azimuths)
    assert np.allclose(angles, measure.viewing_angles)

    # Specifying the hplane param will have the validation step raise
    with pytest.raises(ValueError):
        MultiDistantMeasure.from_viewing_angles(zeniths, azimuths, hplane=0.0)

    # Construct from viewing angles within the same plane using a single azimuth value
    zeniths = np.array([-60, -45, -30, -15, 0, 15, 30, 45, 60])
    azimuths = 0
    measure = MultiDistantMeasure.from_viewing_angles(zeniths, azimuths)
    assert measure.hplane == 0.0 * ureg.deg

    angles = (
        np.reshape(
            np.stack((np.abs(zeniths), [180, 180, 180, 180, 0, 0, 0, 0, 0]), axis=-1),
            (-1, 1, 2),
        )
        * ureg.deg
    )
    assert np.allclose(angles, measure.viewing_angles)

    # Construct an azimuthal ring
    zeniths = 45
    azimuths = np.arange(0, 360, 45)
    measure = MultiDistantMeasure.from_viewing_angles(zeniths, azimuths)
    assert measure.hplane is None

    angles = (
        np.reshape(
            np.stack((np.full_like(azimuths, zeniths), azimuths), axis=-1),
            (-1, 1, 2),
        )
        * ureg.deg
    )
    assert np.allclose(angles, measure.viewing_angles)
