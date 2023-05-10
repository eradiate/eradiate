import mitsuba as mi
import numpy as np

import eradiate.data
from eradiate import KernelContext
from eradiate import unit_registry as ureg
from eradiate.scenes.illumination import SpotIllumination
from eradiate.test_tools.types import check_scene_element


def test_construct_basic(mode_mono):
    # We need a default spectral config
    ctx = KernelContext()

    # Construct without parameters
    illumination = SpotIllumination()
    check_scene_element(illumination, mi_cls=mi.Emitter)

    # Define all parameters
    illumination = SpotIllumination(
        target=[0, 0, 0],
        origin=[1, 1, 1],
        up=[1, 0, 1],
        beam_width=10 * ureg.deg,
        intensity=10,
    )
    check_scene_element(illumination, mi_cls=mi.Emitter)


def test_construct_texture(mode_mono, tmp_path):
    # First we set up a temporary directory with a bitmap file
    from PIL import Image

    filename = tmp_path / "texture.bmp"
    array = np.ones((3, 3, 3))
    im = Image.fromarray(np.uint8(array * 255))
    im.save(filename)

    # We need a default spectral config
    ctx = KernelContext()

    # Construct with custom beam profile filename
    illumination = SpotIllumination(
        target=[0, 0, 0],
        origin=[1, 1, 1],
        up=[1, 0, 1],
        beam_width=10 * ureg.deg,
        intensity=10,
        beam_profile=str(filename),
    )
    check_scene_element(illumination, mi_cls=mi.Emitter)

    # Construct with beam profile from data store
    illumination = SpotIllumination(
        target=[0, 0, 0],
        origin=[1, 1, 1],
        up=[1, 0, 1],
        beam_width=10 * ureg.deg,
        intensity=10,
        beam_profile=eradiate.data.data_store.fetch("textures/gaussian_3sigma.bmp"),
    )
    check_scene_element(illumination, mi_cls=mi.Emitter)


def test_construct_from_size(mode_mono):
    # We need a default spectral config
    ctx = KernelContext()

    spot_radius = 1 * ureg.m
    beam_angle = 3 * ureg.deg

    # spot_radius / tan(beam_angle/2.)
    expected_distance = 38.18846

    illumination = SpotIllumination.from_size_at_target(
        target=[0, 0, 0],
        direction=[0, 0, -1],
        spot_radius=spot_radius,
        beam_width=beam_angle,
    )

    assert np.allclose(illumination.origin.m_as(ureg.m), [0, 0, expected_distance])

    check_scene_element(illumination, mi_cls=mi.Emitter)
