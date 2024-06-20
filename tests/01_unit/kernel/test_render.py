from itertools import product

import mitsuba as mi
import numpy as np

from eradiate import KernelContext
from eradiate.kernel import (
    MitsubaObjectWrapper,
    TypeIdLookupStrategy,
    UpdateMapTemplate,
    UpdateParameter,
    mi_render,
    mi_traverse,
)
from eradiate.spectral.index import SpectralIndex
from eradiate.units import unit_registry as ureg


def test_type_id_lookup_strategy(mode_mono):
    mi_scene = mi.load_dict(
        {
            "type": "scene",
            "bsdf": {"type": "diffuse", "id": "my_bsdf"},
            "rectangle_1": {
                "type": "rectangle",
                "bsdf": {"type": "ref", "id": "my_bsdf"},
            },
            "rectangle_2": {
                "type": "rectangle",
                "bsdf": {"type": "ref", "id": "my_bsdf"},
            },
            "disk_1": {
                "type": "disk",
                "bsdf": {"type": "ref", "id": "my_bsdf"},
            },
            "disk_2": {
                "type": "disk",
                "bsdf": {"type": "diffuse"},
            },
        }
    )

    lookup_strategy = TypeIdLookupStrategy(
        node_type=mi.BSDF, node_id="my_bsdf", parameter_relpath="reflectance.value"
    )

    for shape in mi_scene.shapes():
        path = shape.id()
        assert lookup_strategy(shape.bsdf(), path) == (
            f"{path}.reflectance.value" if path != "disk_2" else None
        )


def test_mi_traverse(mode_mono):
    mi_scene = mi.load_dict(
        {
            "type": "scene",
            "bsdf": {"type": "diffuse", "id": "my_bsdf"},
            "rectangle_1": {
                "type": "rectangle",
                "bsdf": {"type": "ref", "id": "my_bsdf"},
            },
            "rectangle_2": {
                "type": "rectangle",
                "bsdf": {"type": "ref", "id": "my_bsdf"},
            },
            "disk_1": {
                "type": "disk",
                "bsdf": {"type": "ref", "id": "my_bsdf"},
            },
            "disk_2": {
                "type": "disk",
                "bsdf": {"type": "diffuse"},
            },
        }
    )

    umap_template = UpdateMapTemplate(
        {
            "my_bsdf.reflectance.value": UpdateParameter(
                evaluator=lambda x: x,
                flags=UpdateParameter.Flags.ALL,
                lookup_strategy=TypeIdLookupStrategy(
                    node_type=mi.BSDF,
                    node_id="my_bsdf",
                    parameter_relpath="reflectance.value",
                ),
            )
        }
    )

    mi_wrapper = mi_traverse(mi_scene, umap_template)

    # Traversal succeeds
    assert isinstance(mi_wrapper, MitsubaObjectWrapper)

    # Parameter map is correctly extracted
    assert set(mi_wrapper.parameters.keys()) == {
        "my_bsdf.reflectance.value",
        "disk_1.to_world",
        "disk_2.bsdf.reflectance.value",
        "disk_2.to_world",
        "rectangle_1.to_world",
        "rectangle_2.to_world",
    }

    # Parameter ID should be resolved in accordance with the declared lookup strategy
    assert (
        mi_wrapper.umap_template["my_bsdf.reflectance.value"].parameter_id
        == "my_bsdf.reflectance.value"
    )


def test_mi_render(mode_mono):
    mi_scene = mi.load_dict(
        {
            "type": "scene",
            "rectangle": {
                "type": "rectangle",
                "bsdf": {"type": "diffuse", "id": "my_bsdf"},
            },
            "sensor": {
                "type": "distant",
                "film": {"type": "hdrfilm", "width": 1, "height": 1},
                "direction": [0, 0, -1],
                "target": [0, 0, 0],
            },
            "illumination": {
                "type": "directional",
                "direction": [0, 0, -1],
                "irradiance": 1.0,
            },
            "integrator": {"type": "path"},
        }
    )

    umap_template = UpdateMapTemplate(
        {
            "my_bsdf.reflectance.value": UpdateParameter(
                evaluator=lambda ctx: ctx.kwargs["r"],
                flags=UpdateParameter.Flags.ALL,
                lookup_strategy=TypeIdLookupStrategy(
                    node_type=mi.BSDF,
                    node_id="my_bsdf",
                    parameter_relpath="reflectance.value",
                ),
            )
        }
    )

    mi_wrapper = mi_traverse(mi_scene, umap_template)

    reflectances = [0.0, 0.5, 1.0]
    wavelengths = [400.0, 500.0, 600.0] * ureg.nm

    result = mi_render(
        mi_wrapper,
        ctxs=[
            KernelContext(si=SpectralIndex.new(w=w), kwargs={"r": r})
            for (r, w) in zip(reflectances, wavelengths)
        ],
    )

    assert isinstance(result, dict)

    expected = []
    actual = []
    for _, (r, w) in enumerate(zip(reflectances, wavelengths)):
        siah = SpectralIndex.new(w=w).as_hashable
        assert isinstance(result[siah]["sensor"], mi.Bitmap)
        expected.append(r / np.pi)
        actual.append(np.squeeze(result[siah]["sensor"]))

    np.testing.assert_allclose(actual, expected)


def test_mi_render_multisensor(mode_mono):
    mi_scene = mi.load_dict(
        {
            "type": "scene",
            "rectangle": {
                "type": "rectangle",
                "bsdf": {"type": "diffuse", "id": "my_bsdf"},
            },
            "sensor1": {
                "type": "distant",
                "film": {"type": "hdrfilm", "width": 1, "height": 1},
                "direction": [0, 0, -1],
                "target": [0, 0, 0],
            },
            "sensor2": {
                "type": "distant",
                "film": {"type": "hdrfilm", "width": 1, "height": 1},
                "direction": [0, 0, -1],
                "target": [0, 0, 0],
            },
            "illumination": {
                "type": "directional",
                "direction": [0, 0, -1],
                "irradiance": 1.0,
            },
            "integrator": {"type": "path"},
        }
    )

    umap_template = UpdateMapTemplate(
        {
            "my_bsdf.reflectance.value": UpdateParameter(
                evaluator=lambda ctx: ctx.kwargs["r"],
                flags=UpdateParameter.Flags.ALL,
                lookup_strategy=TypeIdLookupStrategy(
                    node_type=mi.BSDF,
                    node_id="my_bsdf",
                    parameter_relpath="reflectance.value",
                ),
            )
        }
    )

    mi_wrapper = mi_traverse(mi_scene, umap_template)

    reflectances = [0.0, 0.5, 1.0]
    wavelengths = [400.0, 500.0, 600.0] * ureg.nm

    result = mi_render(
        mi_wrapper,
        ctxs=[
            KernelContext(si=SpectralIndex.new(w=w), kwargs={"r": r})
            for (r, w) in zip(reflectances, wavelengths)
        ],
    )

    # The result must be a nested dict with one level-one element per wavelength,
    # and one level-two element per sensor
    assert isinstance(result, dict)
    assert np.allclose(
        list(result.keys()),
        [w.m for w in wavelengths],
    )

    sensors_keys = set()
    for spectral_key in result.keys():
        sensors_keys.update(set(result[spectral_key].keys()))
    assert sensors_keys == {"sensor1", "sensor2"}
    assert all(
        isinstance(result[spectral_key][sensor_key], mi.Bitmap)
        for (spectral_key, sensor_key) in product(result.keys(), sensors_keys)
    )
