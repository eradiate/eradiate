import attrs
import matplotlib.pyplot as plt
import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.attrs import AUTO
from eradiate.grid import PlaneParallelGridCoords
from eradiate.scenes.atmosphere import (
    HeterogeneousAtmosphere,
    MolecularAtmosphere,
)
from eradiate.scenes.geometry import PlaneParallelGeometry
from eradiate.test_tools.regression import RegressionTestFailure, ZTest
from eradiate.test_tools.report import figure_to_html, report_logger
from eradiate.test_tools.test_cases import rami4atm

# Small enough that oblique rays (these cases go out to 75 deg zenith) need to
# sample outside the grid's own horizontal extent, so wrap_mode=CLAMP is
# actually exercised. The atmosphere stays horizontally uniform, so a correct
# CLAMP still reproduces the 1D (infinite homogeneous slab) reference.
GRID_EXTENT = 2.0 * ureg.km

cases = [
    "hom00_rpv_e00s_m04_z30a000_brfpp",
    "hom00_whi_s00s_m04_z30a000_brfpp",
    "hom00_rpv_0d6s_m04_z30a000_brfpp",
    "hom00_rpv_sd2s_m04_z30a000_brfpp",
    "hom00_rpv_ac2s_m04_z30a000_brfpp",
    "hom00_rpv_ec6s_m04_z30a000_brfpp",
]


@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.parametrize("case", cases)
def test_rami4atm_gridded(mode_ckd_double, case):
    case_obj = rami4atm.CASES[case]
    variables = ["radiance"]
    threshold = 0.005

    exps = case_obj.make_experiments(spp=1000)
    exps_3d = []

    resolution = (2, 3)
    resx, resy = resolution

    for exp in exps:
        mol_atm_1d = exp.atmosphere.molecular_atmosphere

        geometry = PlaneParallelGeometry(
            grid=PlaneParallelGridCoords.from_extent_and_resolution(
                levels=exp.geometry.grid.levels,
                extent_x=GRID_EXTENT,
                extent_y=GRID_EXTENT,
                n_cells_x=resx,
                n_cells_y=resy,
            ),
        )

        mol_atm_3d = None
        if mol_atm_1d:
            mol_atm_3d = MolecularAtmosphere(
                geometry=geometry,
                absorption_data=mol_atm_1d.absorption_data,
                thermoprops=mol_atm_1d.thermoprops,
                has_absorption=mol_atm_1d.has_absorption,
                has_scattering=mol_atm_1d.has_scattering,
            )

        atm = HeterogeneousAtmosphere(
            geometry=geometry,
            molecular_atmosphere=mol_atm_3d,
            particle_ensembles=[
                attrs.evolve(
                    ensemble,
                    tau_ref=ensemble.tau_ref * np.ones(resolution),
                    geometry=geometry,
                )
                for ensemble in exp.atmosphere.particle_ensembles
            ],
        )

        exps_3d.append(
            attrs.evolve(
                exp,
                atmosphere=atm,
                geometry=geometry,
                integrator={"type": "volpath", "moment": True},
                background_spectral_grid=AUTO,
            )
        )

    raw_results_3d = [eradiate.run(exp) for exp in exps_3d]

    result = case_obj.postprocess(raw_results_3d)
    report_logger.html(result._repr_html_())

    raw_results = [eradiate.run(exp) for exp in exps]
    reference = case_obj.postprocess(raw_results)

    report_logger.html(reference._repr_html_())

    for variable in variables:
        test = ZTest(threshold, variable=variable)
        outcome = test.evaluate(result, reference)

        figure, _ = test.plot(result, reference, outcome)
        try:
            report_logger.html(figure_to_html(figure))
        finally:
            plt.close(figure)

        report_logger.info(str(outcome))

        if not outcome.passed:
            raise RegressionTestFailure(outcome)
