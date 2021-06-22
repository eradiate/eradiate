import enoki as ek
import numpy as np
import pytest
from matplotlib import pyplot as plt

from eradiate import unit_context_config as ucc
from eradiate import unit_context_kernel as uck
from eradiate import unit_registry as ureg
from eradiate.scenes.core import KernelDict
from eradiate.scenes.measure._distant import (
    DistantRadianceMeasure,
    TargetOrigin,
    TargetOriginPoint,
    TargetOriginRectangle,
    TargetOriginSphere,
)


def test_target_origin(mode_mono):
    from mitsuba.core import Point3f

    # TargetOriginPoint: basic constructor
    with ucc.override({"length": "km"}):
        t = TargetOriginPoint([0, 0, 0])
        assert t.xyz.units == ureg.km

    with pytest.raises(ValueError):
        TargetOriginPoint(0)

    # TargetOriginPoint: check kernel item
    with ucc.override({"length": "km"}), uck.override({"length": "m"}):
        t = TargetOriginPoint([1, 2, 0])
        assert ek.allclose(t.kernel_item(), [1000, 2000, 0])

    # TargetOriginRectangle: basic constructor
    with ucc.override({"length": "km"}):
        t = TargetOriginRectangle(0, 1, 0, 1)
        assert t.xmin == 0.0 * ureg.km
        assert t.xmax == 1.0 * ureg.km
        assert t.ymin == 0.0 * ureg.km
        assert t.ymax == 1.0 * ureg.km

    with ucc.override({"length": "m"}):
        t = TargetOriginRectangle(0, 1, 0, 1)
        assert t.xmin == 0.0 * ureg.m
        assert t.xmax == 1.0 * ureg.m
        assert t.ymin == 0.0 * ureg.m
        assert t.ymax == 1.0 * ureg.m

    with pytest.raises(ValueError):
        TargetOriginRectangle(0, 1, "a", 1)

    with pytest.raises(ValueError):
        TargetOriginRectangle(0, 1, 1, -1)

    # TargetOriginRectangle: check kernel item
    t = TargetOriginRectangle(-1, 1, -1, 1)

    with uck.override({"length": "mm"}):  # Tricky: we can't compare transforms directly
        kernel_item = t.kernel_item()["to_world"]
        assert ek.allclose(
            kernel_item.transform_point(Point3f(-1, -1, 0)), [-1000, -1000, 0]
        )
        assert ek.allclose(
            kernel_item.transform_point(Point3f(1, 1, 0)), [1000, 1000, 0]
        )
        assert ek.allclose(
            kernel_item.transform_point(Point3f(1, 1, 42)), [1000, 1000, 42]
        )

    # TargetOriginSphere: basic constructor
    with ucc.override({"length": "km"}):
        t = TargetOriginSphere([0, 0, 0], 1)
        assert t.center.units == ureg.km
        assert t.radius.units == ureg.km

    with pytest.raises(ValueError):
        TargetOriginSphere(0, 1)
    with pytest.raises(ValueError):
        TargetOriginSphere([0, 0, 0], -1)

    # TargetOriginSphere: check kernel item
    t = TargetOriginSphere([0, 0, 1], 1)
    with uck.override({"length": "mm"}):
        kernel_item = t.kernel_item()
        assert set(kernel_item.keys()) == {"type", "radius", "center"}
        assert kernel_item["type"] == "sphere"
        assert np.allclose(kernel_item["center"], [0, 0, 1000])
        assert np.allclose(kernel_item["radius"], 1000)

    # Factory: basic test
    with ucc.override({"length": "m"}):
        t = TargetOrigin.new("point", xyz=[1, 1, 0])
        assert isinstance(t, TargetOriginPoint)
        assert np.allclose(t.xyz, ureg.Quantity([1, 1, 0], ureg.m))

        t = TargetOrigin.new("rectangle", 0, 1, 0, 1)
        assert isinstance(t, TargetOriginRectangle)

    # Converter: basic test
    with ucc.override({"length": "m"}):
        t = TargetOrigin.convert({"type": "point", "xyz": [1, 1, 0]})
        assert isinstance(t, TargetOriginPoint)
        assert np.allclose(t.xyz, ureg.Quantity([1, 1, 0], ureg.m))

        t = TargetOrigin.convert([1, 1, 0])
        assert isinstance(t, TargetOriginPoint)
        assert np.allclose(t.xyz, ureg.Quantity([1, 1, 0], ureg.m))

        with pytest.raises(ValueError):
            TargetOrigin.convert({"xyz": [1, 1, 0]})


def test_distant_radiance(mode_mono):
    # Test default constructor
    d = DistantRadianceMeasure()
    assert KernelDict.new(d).load() is not None

    # Test target support
    # -- Target a point
    d = DistantRadianceMeasure(target=[0, 0, 0])
    assert KernelDict.new(d).load() is not None

    # -- Target an axis-aligned rectangular patch
    d = DistantRadianceMeasure(
        target={"type": "rectangle", "xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    )
    assert KernelDict.new(d).load() is not None

    # Test origin support
    # -- Project origins to a sphere
    d = DistantRadianceMeasure(
        origin={"type": "sphere", "center": [0, 0, 0], "radius": 1}
    )
    assert KernelDict.new(d).load() is not None


def test_distant_radiance_postprocessing(mode_mono):
    # We use a peculiar rectangular film size to make sure that we get dimensions
    # right
    d = DistantRadianceMeasure(film_resolution=(32, 16))

    # Add test data to results
    d.results.raw = {
        550.0: {"values": {"sensor": np.ones((16, 32, 1))}, "spp": {"sensor": 128}}
    }

    # Postprocessing succeeds and viewing angles have correct bounds
    ds = d.postprocess()
    assert "vza" in ds.coords
    assert np.allclose(ds.vza.min(), 5.06592926)  # Value calculated manually
    assert np.allclose(ds.vza.max(), 86.47273911)  # Value calculated manually
    assert "vaa" in ds.coords
    assert np.allclose(ds.vaa.min(), -177.09677419)  # Value calculated manually
    assert np.allclose(ds.vaa.max(), 177.09677419)  # Value calculated manually

    # We now move on to the plane case
    d._film_resolution = (32, 1)
    # Mismatched film size and raw data dimensions raises
    with pytest.raises(ValueError):
        d.postprocess()

    # Postprocessing succeeds and viewing angles have correct bounds
    d.results.raw = {
        550.0: {"values": {"sensor": np.ones((1, 32, 1))}, "spp": {"sensor": 128}}
    }
    ds = d.postprocess()
    assert "vza" in ds.coords
    assert np.allclose(ds.vza.min(), -87.1875)  # Value manually calculated
    assert np.allclose(ds.vza.max(), 87.1875)  # Value manually calculated
    assert "vaa" in ds.coords
    assert np.allclose(ds.vaa, 0.0)
