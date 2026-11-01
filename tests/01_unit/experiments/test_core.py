import warnings
from typing import Any

import numpy as np
import pytest
import xarray as xr

import eradiate
from eradiate.experiments import AtmosphereExperiment, EarthObservationExperiment
from eradiate.experiments._core import MeasureRegistry
from eradiate.scenes.core import SceneElement
from eradiate.spectral import BandSRF
from eradiate.units import unit_registry as ureg


class ConcreteEarthObservationExperiment(EarthObservationExperiment):
    def _context_kwargs(self) -> dict[str, Any]:
        return {}

    @property
    def scene_objects(self) -> dict[str, SceneElement]:
        return {measure.id: measure for measure in self.measures}


def test_contexts(mode_mono):
    """
    Check if generated kernel contexts account for selected sensors appropriately.
    """
    exp = ConcreteEarthObservationExperiment(
        measures=[
            {
                "type": "mdistant",
                "id": mes_id,
                "srf": {"type": "delta", "wavelengths": w},
            }
            for mes_id, w in [
                ("mes1", [440.0]),
                ("mes2", [550.0]),
                ("mes3", [440.0, 550.0]),
                ("mes4", [550.0, 660.0]),
            ]
        ]
    )

    # Check initialization context
    assert np.allclose(exp.context_init().si.w.m, 440.0)

    # We need to init the scene so that mapping measures to sensors is possible
    exp.init()

    # Check contexts generated for each measure configuration
    for measures, expected in [
        (None, [(440.0, [0, 2]), (550.0, [1, 2, 3]), (660.0, [3])]),
        (0, [(440.0, [0])]),
        ([0, 1], [(440.0, [0]), (550.0, [1])]),
        ([0, 1, 2], [(440.0, [0, 2]), (550.0, [1, 2])]),
        ([1, 3], [(550.0, [1, 3]), (660.0, [3])]),
        ([2, 3], [(440.0, [2]), (550.0, [2, 3]), (660.0, [3])]),
    ]:
        result = exp.contexts(measures)
        assert len(result) == len(expected)
        for ctx, (w, sensors) in zip(result, expected):
            assert np.isclose(ctx.si.w.m, w), f"{measures = }, {w = }"
            assert ctx.active_sensors == sensors, f"{measures = }, {w = }"


@pytest.fixture()
def srf():
    return {"type": "delta", "wavelengths": [540, 550] * ureg.nm}


def test_measure_registry(mode_mono, srf):
    m = MeasureRegistry(
        [{"type": "mdistant", "id": f"mdistant_{i}", "srf": srf} for i in range(3)]
    )

    assert m.get_id(2) == "mdistant_2"
    assert m.get_id("mdistant_2") == "mdistant_2"
    assert m.get_index(2) == 2
    assert m.get_index("mdistant_2") == 2

    assert m.resolve(2) is m[2]
    assert m.resolve("mdistant_2") is m[2]


def test_measure_registry_fail(mode_mono, srf):
    # Detect duplicate measure IDs
    with pytest.raises(ValueError):
        MeasureRegistry(
            [{"type": "mdistant", "id": "mdistant", "srf": srf} for _ in range(3)]
        )


@pytest.fixture(scope="function")
def atmosphere_experiment():
    yield AtmosphereExperiment(
        atmosphere=None,
        measures=[
            {
                "type": "mdistant",
                "id": f"mdistant_{i}",
                "srf": {"type": "delta", "wavelengths": [550.0]},
            }
            for i in range(3)
        ],
    )


@pytest.mark.parametrize(
    "measures, expected_type",
    [
        (None, dict),
        (1, xr.Dataset),
        ([0, 1], dict),
    ],
)
def test_run_function(modes_all_double, atmosphere_experiment, measures, expected_type):
    result = eradiate.run(atmosphere_experiment, measures=measures, spp=64)
    assert isinstance(result, expected_type)


def test_run_function_multiple_times(mode_mono, atmosphere_experiment):
    result = eradiate.run(atmosphere_experiment, measures=0, spp=4)
    assert isinstance(result, xr.Dataset)
    result = eradiate.run(atmosphere_experiment, measures=1, spp=4)
    assert isinstance(result, xr.Dataset)
    assert len(atmosphere_experiment.results) == 2


@pytest.mark.usefixtures("mode_ckd")
class TestContextsSpp:
    """Sample counts attached to kernel contexts."""

    @pytest.fixture
    def experiment(self):
        band = BandSRF(wavelengths=[540.0, 550.0, 560.0], values=[0.0, 1.0, 0.0])
        delta = {"type": "delta", "wavelengths": [550.0]}
        exp = AtmosphereExperiment(
            atmosphere=None,
            measures=[
                {"type": "mdistant", "id": "band", "srf": band, "spp": 1000},
                {
                    "type": "mdistant",
                    "id": "stratified",
                    "srf": delta,
                    "spp": 100,
                    "sampler": "stratified",
                },
                {
                    "type": "mdistant",
                    "id": "uniform",
                    "srf": delta,
                    "spp": 7,
                    "spp_allocation": "uniform",
                },
            ],
        )
        exp.init()
        return exp

    def test_contexts(self, experiment):
        ctxs = experiment.contexts()
        spp = {i: [ctx.spp[i] for ctx in ctxs if i in ctx.spp] for i in range(3)}

        # Spectral indices shared by measures hold the counts of every sensor
        assert any(len(ctx.spp) == 3 for ctx in ctxs)
        assert all(ctx.active_sensors == list(ctx.spp) for ctx in ctxs)
        # Weighted, BandSRF: a band total
        assert sum(spp[0]) == 1000
        assert min(spp[0]) >= 2  # Default Measure.spp_min
        # Stratified sampler: rounded up to square numbers
        assert all(int(np.sqrt(n)) ** 2 == n for n in spp[1])
        assert sum(spp[1]) >= 100
        # Uniform policy (overriding the setting): full target everywhere
        assert spp[2] == [7] * len(spp[2])

    def test_process(self, experiment, monkeypatch):
        """Results record the sample counts and policy actually used."""
        monkeypatch.setitem(eradiate.config.settings, "SAMPLE_ALLOCATION", "weighted")
        ctxs = {ctx.si.as_hashable: ctx for ctx in experiment.contexts([1])}
        result = eradiate.run(experiment, measures=1)

        measure = experiment.measures[1]
        for si_hash, imgs in measure.mi_results.items():
            assert imgs["spp"] == ctxs[si_hash].spp[1]
        assert result.attrs["sample_allocation"] == "weighted"

        # Changing the setting after processing does not alter the record
        monkeypatch.setitem(eradiate.config.settings, "SAMPLE_ALLOCATION", "uniform")
        experiment.postprocess(1)
        assert experiment.results["stratified"].attrs["sample_allocation"] == "weighted"

    @pytest.mark.parametrize("spp_min", [1, 2])
    def test_spp_min(self, spp_min):
        """
        With every iteration at ``spp_min`` samples, a value of 1 leaves the
        band variance undefined (and warns), 2 does not.
        """
        band = BandSRF(wavelengths=[540.0, 550.0, 560.0], values=[0.0, 1.0, 0.0])
        exp = AtmosphereExperiment(
            atmosphere=None,
            integrator={"type": "path", "moment": True},
            measures={"type": "mdistant", "srf": band, "spp_min": spp_min},
        )
        exp.init()
        exp.measures[0].spp = spp_min * sum(len(q.nodes) for q in exp.ckd_quads[0])

        if spp_min == 1:
            with pytest.warns(UserWarning, match="single sample"):
                result = eradiate.run(exp)
            assert np.all(np.isnan(result.radiance_srf_var.values))
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                result = eradiate.run(exp)
            assert np.all(np.isfinite(result.radiance_srf_var.values))


def test_init_updates_ckd_quads(mode_ckd):
    """Quadrature rules follow a CKD quadrature configuration set after construction."""
    exp = AtmosphereExperiment(atmosphere=None)
    exp.ckd_quad_config = {"type": "gauss_legendre", "ng_max": 4, "policy": "fixed"}
    exp.init()

    assert all(len(quad.nodes) == 4 for quad in exp.ckd_quads[0])
    assert len(list(exp.spectral_indices(0))) == 4 * len(exp.ckd_quads[0])
