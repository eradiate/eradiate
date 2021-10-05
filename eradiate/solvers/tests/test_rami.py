import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.exceptions import ModeError
from eradiate.scenes.biosphere import DiscreteCanopy
from eradiate.scenes.measure import DistantRadianceMeasure
from eradiate.solvers.rami import RamiScene, RamiSolverApp


@pytest.mark.slow
def test_rami_scene(mode_mono):
    ctx = KernelDictContext()

    # Construct with default parameters
    s = RamiScene()
    assert s.kernel_dict(ctx=ctx).load() is not None

    # Test non-trivial init sequence steps

    # -- Init with a single measure (not wrapped in a sequence)
    s = RamiScene(measures=DistantRadianceMeasure())
    assert s.kernel_dict(ctx=ctx).load() is not None
    # -- Init from a dict-based measure spec
    # ---- Correctly wrapped in a sequence
    s = RamiScene(measures=[{"type": "distant_radiance"}])
    assert s.kernel_dict(ctx=ctx).load() is not None
    # ---- Not wrapped in a sequence
    s = RamiScene(measures={"type": "distant_radiance"})
    assert s.kernel_dict(ctx=ctx).load() is not None

    # -- Surface size is appropriately overridden with canopy size
    s = RamiScene(
        canopy=DiscreteCanopy.homogeneous(
            lai=3.0,
            leaf_radius=0.1 * ureg.m,
            l_horizontal=10.0 * ureg.m,
            l_vertical=2.0 * ureg.m,
        )
    )
    ctx = KernelDictContext()
    kernel_scene = s.kernel_dict(ctx)
    assert np.allclose(
        kernel_scene["surface"]["to_world"].transform_point([1, -1, 0]),
        [5, -5, 0],
    )

    # -- Distant sensor target zone is appropriately defined
    s = RamiScene(
        canopy=DiscreteCanopy.homogeneous(
            lai=3.0,
            leaf_radius=0.1 * ureg.m,
            l_horizontal=10.0 * ureg.m,
            l_vertical=2.0 * ureg.m,
        ),
        measures=DistantRadianceMeasure(),
    )
    target = s.measures[0].target
    canopy = s.canopy
    assert np.isclose(target.xmin, -0.5 * canopy.size[0])
    assert np.isclose(target.xmax, 0.5 * canopy.size[0])
    assert np.isclose(target.ymin, -0.5 * canopy.size[1])
    assert np.isclose(target.ymax, 0.5 * canopy.size[1])


def test_rami_solver_app_new():
    # Test the new() constructor wrapper

    # Should raise if no mode is set
    with pytest.raises(ModeError):
        eradiate.set_mode("none")
        with pytest.warns(DeprecationWarning):
            RamiSolverApp.new()

    # Should successfully construct a OneDimSolver otherwise
    for mode in ("mono", "mono_double"):
        eradiate.set_mode(mode)
        with pytest.warns(DeprecationWarning):
            RamiSolverApp.new()


def test_rami_scene_real_life(mode_mono):
    ctx = KernelDictContext()

    # Construct with typical parameters
    s = RamiScene(
        surface={"type": "lambertian"},
        canopy={
            "type": "discrete_canopy",
            "construct": "homogeneous",
            "lai": 3.0,
            "leaf_radius": 0.1 * ureg.m,
            "l_horizontal": 10.0 * ureg.m,
            "l_vertical": 2.0 * ureg.m,
        },
        illumination={"type": "directional", "zenith": 45.0},
        measures={"type": "distant_reflectance"},
    )
    assert s.kernel_dict(ctx=ctx).load() is not None


@pytest.mark.slow
def test_rami_solver_app_run(mode_mono):
    """Test the creation of a DataArray from the solver result

    We create a default scene with a set of zenith and azimuth angles,
    render the scene and create the DataArray.

    We assert the correct setting of the DataArray coordinates and dimensions,
    as well as the correct setting of data.
    """
    with pytest.warns(DeprecationWarning):
        app = RamiSolverApp(
            scene=RamiScene(
                measures=[
                    {
                        "type": "distant_reflectance",
                        "id": "toa_hsphere",
                        "film_resolution": (32, 32),
                        "spp": 1000,
                    },
                ]
            )
        )

    app.run()

    results = app.results["toa_hsphere"]

    # Post-processing creates expected variables ...
    assert set(results.data_vars) == {"irradiance", "brf", "brdf", "lo", "spp"}
    # ... dimensions
    assert set(results["lo"].dims) == {"sza", "saa", "x", "y", "w", "channel"}
    assert set(results["irradiance"].dims) == {"sza", "saa", "w"}
    # ... and other coordinates
    assert set(results["lo"].coords) == {
        "sza",
        "saa",
        "vza",
        "vaa",
        "x",
        "y",
        "w",
        "channel",
    }
    assert set(results["irradiance"].coords) == {"sza", "saa", "w"}

    # We just check that we record something as expected
    assert np.all(results["lo"].data > 0.0)
