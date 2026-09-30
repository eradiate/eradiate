"""
This benchmark sweeps the spectral dimension of an atmosphere experiment
without calling Mitsuba: it generates kernel contexts and evaluates the scene
parameter update map for each of them. It estimates the Python-side overhead
of the spectral loop.
"""

import time
from contextlib import contextmanager

import numpy as np

import eradiate
from eradiate.contexts import KernelContext
from eradiate.experiments import AtmosphereExperiment
from eradiate.scenes.core import traverse


@contextmanager
def timer(label: str = "Elapsed"):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.3f}s")


eradiate.set_mode("ckd")

exp = AtmosphereExperiment(
    geometry={"type": "plane_parallel", "zgrid": np.linspace(0, 120e3, 12001)},
    atmosphere={"type": "molecular", "absorption_data": "panellus"},
    measures={
        "type": "mdistant",
        "construct": "hplane",
        "azimuth": 30.0,
        "zeniths": [-75.0],
        "srf": {"type": "uniform", "wmin": 525.0, "wmax": 575.0},
        "spp": 1,
    },
)

# ponytail: no kernel scene, so parameter lookups keep their template keys
with timer("Scene traversal"):
    _, umap_template = traverse(exp.scene)
    umap_template.update(exp.kpmap)

with timer("Context generation"):
    kwargs = exp._context_kwargs()
    ctxs = [KernelContext(si, kwargs=kwargs) for si in exp.spectral_indices(0)]
print(f"Contexts: {len(ctxs)}")

with timer("Parameter map evaluation"):
    for ctx in ctxs:
        umap_template.render(ctx)
