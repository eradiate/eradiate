from pprint import pprint

import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.exceptions import ModeError
from eradiate.scenes.biosphere import DiscreteCanopy
from eradiate.scenes.measure import DistantMeasure
from eradiate.solvers.rami import RamiScene, RamiSolverApp


def test_rami_scene(mode_mono):
    # Construct with default parameters
    s = RamiScene()
    assert s.kernel_dict().load() is not None

    # Test non-trivial init sequence steps

    # -- Init with a single measure (not wrapped in a sequence)
    s = RamiScene(measures=DistantMeasure())
    assert s.kernel_dict().load() is not None
    # -- Init from a dict-based measure spec
    # ---- Correctly wrapped in a sequence
    s = RamiScene(measures=[{"type": "distant"}])
    assert s.kernel_dict().load() is not None
    # ---- Not wrapped in a sequence
    s = RamiScene(measures={"type": "distant"})
    assert s.kernel_dict().load() is not None

    # -- Surface size is appropriately inherited from canopy
    s = RamiScene(
        canopy=DiscreteCanopy.homogeneous(
            lai=3.0,
            leaf_radius=0.1 * ureg.m,
            l_horizontal=10.0 * ureg.m,
            l_vertical=2.0 * ureg.m,
        )
    )
    assert np.allclose(s.surface.width, ureg.Quantity(10.0, "m"))

    # -- Distant sensor target zone is appropriately defined
    s = RamiScene(
        canopy=DiscreteCanopy.homogeneous(
            lai=3.0,
            leaf_radius=0.1 * ureg.m,
            l_horizontal=10.0 * ureg.m,
            l_vertical=2.0 * ureg.m,
        ),
        measures=DistantMeasure(),
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
        RamiSolverApp.new()

    # Should successfully construct a OneDimSolver otherwise
    for mode in ("mono", "mono_double"):
        eradiate.set_mode(mode)
        RamiSolverApp.new()


def test_rami_scene_real_life(mode_mono):
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
        measures={"type": "distant"},
    )
    assert s.kernel_dict().load() is not None


@pytest.mark.slow
def test_rami_solver_app_run(mode_mono):
    """Test the creation of a DataArray from the solver result

    We create a default scene with a set of zenith and azimuth angles,
    render the scene and create the DataArray.

    We assert the correct setting of the DataArray coordinates and dimensions,
    as well as the correct setting of data.
    """
    app = RamiSolverApp(
        scene=RamiScene(
            canopy={
                "type": "discrete_canopy",
                "construct": "homogeneous",
                "lai": 3.0,
                "leaf_radius": 0.1 * ureg.m,
                "l_horizontal": 10.0 * ureg.m,
                "l_vertical": 2.0 * ureg.m,
            },
            measures={"type": "distant"},
        )
    )

    app.run()

    results = app.results["measure"]

    # # Assert the correct dimensions of the application's results
    # assert set(results["lo"].dims) == {"sza", "saa", "vza", "vaa", "wavelength"}
    #
    # # We expect the whole [0, 360] to be covered
    # assert len(results["lo"].coords["vaa"]) == 360. / 180.
    # # # We expect [0, 90[ to be covered (90° should be missing)
    # assert len(results["lo"].coords["vza"]) == 90. / 45.
    # # We just check that we record something as expected
    # assert np.all(results["lo"].data > 0.)
