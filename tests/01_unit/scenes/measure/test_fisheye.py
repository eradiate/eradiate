import mitsuba as mi
import numpy as np
import pytest

import eradiate
from eradiate import KernelContext
from eradiate.experiments import AtmosphereExperiment
from eradiate.scenes.core import traverse
from eradiate.scenes.measure import FisheyeCameraMeasure
from eradiate.test_tools.types import check_scene_element

# A Sigma 4.5 mm F2.8 calibration sheet: pixel values at 6000 x 4000
CALIBRATION = {
    "projection_model": "polynomial",
    "lens_coefficients": [0.7195, -0.0637],
    "center_x": 3017,
    "center_y": 1989,
    "image_circle_radius": 1643,
    "calibration_resolution": (6000, 4000),
    "film_resolution": (600, 400),
}


@pytest.mark.parametrize(
    "tested",
    [
        {},
        {"projection_model": "equidistant"},
        {"projection_model": "stereographic"},
        {"projection_model": "orthographic"},
        {"projection_model": "equisolid_full"},
        {"projection_model": "polynomial", "lens_coefficients": [0.7195]},
        {
            "projection_model": "equisolid",
            "image_circle_radius": 0.32 * 32,
            "center_x": 0.36 * 32,
            "center_y": 0.60 * 32,
        },
        CALIBRATION,
    ],
    ids=[
        "no_args",
        "equidistant",
        "stereographic",
        "orthographic",
        "equisolid_full",
        "polynomial",
        "decentred_analytic",
        "polynomial_calibrated",
    ],
)
def test_fisheye_construct(mode_mono, tested):
    measure = FisheyeCameraMeasure(**tested)
    check_scene_element(measure, mi.Sensor)


def test_fisheye_construct_invalid(mode_mono):
    for kwargs in [
        {"origin": [0, 0, 0], "target": [0, 0, 0]},  # Origin and target coincide
        {"origin": [0, 0, 0], "target": [0, 0, 1]},  # up colinear with direction
        {"projection_model": "bogus"},  # Unknown projection model
        {"projection_model": "polynomial"},  # Polynomial without coefficients
        {  # Coefficients are the one field that needs the polynomial model
            "projection_model": "equisolid",
            "lens_coefficients": [0.7195],
        },
        {**CALIBRATION, "film_resolution": (32, 32)},  # Aspect ratio mismatch
        {"film_resolution": (0, 32)},  # Degenerate film
    ]:
        with pytest.raises(ValueError):
            FisheyeCameraMeasure(**kwargs)


def test_fisheye_kernel_dict(mode_mono):
    template, _ = traverse(FisheyeCameraMeasure(**CALIBRATION))
    kdict = template.render(ctx=KernelContext())

    assert kdict["projection_model"] == "polynomial"
    assert kdict["lens_coefficients"] == "0.7195, -0.0637"
    assert kdict["center_x"] == pytest.approx(3017 / 6000)
    assert kdict["center_y"] == pytest.approx(1989 / 4000)
    assert kdict["radius"] == pytest.approx(1643 / 6000)
    assert kdict["fov"] == 180.0

    # The analytic models forward no calibration at all
    template, _ = traverse(FisheyeCameraMeasure(projection_model="equisolid"))
    kdict = template.render(ctx=KernelContext())
    for key in ["lens_coefficients", "center_x", "center_y", "radius"]:
        assert key not in kdict


def test_fisheye_valid_mask(mode_mono):
    # valid_mask reproduces the plugin's in-circle test in NumPy so that results
    # can carry it, so it has to be pinned against the kernel.
    measure = FisheyeCameraMeasure(
        projection_model="polynomial",
        lens_coefficients=[0.7195, -0.0637],
        center_x=0.36 * 32,
        center_y=0.60 * 32,
        image_circle_radius=0.32 * 32,
        film_resolution=(32, 32),
    )
    template, _ = traverse(measure)
    sensor = mi.load_dict(template.render(ctx=KernelContext()))

    # The kernel's own answer: a sample imaging no direction gets zero weight
    n = 32
    expected = np.empty((n, n), dtype=bool)
    for j, i in np.ndindex(n, n):
        _, weight = sensor.sample_ray(
            0.0, 0.5, [(i + 0.5) / n, (j + 0.5) / n], [0.5, 0.5]
        )
        expected[j, i] = np.any(np.asarray(weight) != 0.0)

    assert np.array_equal(measure.valid_mask, expected)


def test_fisheye_medium(mode_mono):
    measure = FisheyeCameraMeasure()
    template, _ = traverse(measure)

    kdict = template.render(ctx=KernelContext())
    assert "medium" not in kdict

    kdict = template.render(
        ctx=KernelContext(kwargs={"measure.atmosphere_medium_id": "test_atmosphere"})
    )
    assert kdict["medium"] == {"type": "ref", "id": "test_atmosphere"}


def test_fisheye_full_scene(mode_mono):
    # Smoke test running the full processing chain, pointed down at a Lambertian
    # surface: reflectance 1 under irradiance pi reflects a radiance of 1. The
    # film is not square, so a dropped transposition fails loudly.
    measure = FisheyeCameraMeasure(
        origin=[0, 0, 0.5], target=[0, 0, 0], up=[0, 1, 0], film_resolution=(24, 16)
    )
    exp = AtmosphereExperiment(
        atmosphere=None,
        illumination={"type": "directional", "irradiance": np.pi},
        surface={"type": "lambertian", "reflectance": 1.0},
        measures=[measure],
    )
    result = eradiate.run(exp, spp=1)
    radiance = result.radiance.values.squeeze()

    # Rim pixels are only partly covered by the image circle, so assert on the
    # interior, the leftmost columns are outside it entirely
    np.testing.assert_allclose(radiance[4:12, 8:16], 1.0)
    np.testing.assert_allclose(radiance[:, :4], 0.0)

    # The fisheye also publishes a validity mask and describes its projection
    assert result.valid.dims == ("y_index", "x_index")
    assert result.valid.dtype == bool
    assert np.array_equal(result.valid, measure.valid_mask)
    assert result.attrs["projection_model"] == "equisolid"
