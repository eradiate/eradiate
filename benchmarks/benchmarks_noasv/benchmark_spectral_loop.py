"""
This benchmark is an empty simulation that sweeps the spectral dimension to
estimate the amount of time spent on evaluating the spectral loop.
"""

import time
from contextlib import contextmanager

import numpy as np

import eradiate
from eradiate.experiments import AtmosphereExperiment
from eradiate.quad import Quad
from eradiate.units import unit_registry as ureg


@contextmanager
def timer(label: str = "Elapsed"):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.3f}s")


eradiate.set_mode("ckd")

try:
    from eradiate.radprops import absdb_factory

    db = absdb_factory.create("panellus")
except ImportError:
    from eradiate.radprops import AbsorptionDatabase

    db = AbsorptionDatabase.from_name("panellus")

quad = Quad.gauss_legendre(16)
gs = quad.nodes
ws = np.array([x[1] for x in db._spectral_coverage.index.values]) / ureg.nm

exp = AtmosphereExperiment(
    geometry={"type": "plane_parallel", "zgrid": np.linspace(0, 120e3, 12001)},
    atmosphere={"type": "molecular", "absorption_data": "panellus"},
    measures={
        "type": "mdistant",
        "construct": "hplane",
        "azimuth": 30.0,
        "zeniths": np.linspace(-75, 76, 1),
        "srf": {"type": "uniform", "wmin": 525.0, "wmax": 575.0},
        "spp": 1,
    },
)
exp.init()


with timer("Simulation"):
    exp.process()
