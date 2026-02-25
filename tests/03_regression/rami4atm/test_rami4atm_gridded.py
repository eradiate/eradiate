import attrs
import pytest
from robot.api import logger

import eradiate
from eradiate import fresolver
from eradiate.scenes.atmosphere import (
    GriddedHeterogeneousAtmosphere,
    GriddedMolecularAtmosphere,
)
from eradiate.test_tools.regression import SidakTTest
from eradiate.test_tools.test_cases import rami4atm

cases = [
    "hom00_rpv_e00s_m04_z30a000_brfpp",
    "hom00_whi_s00s_m04_z30a000_brfpp",
]


@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.parametrize("case", cases)
@pytest.mark.filterwarnings(
    "ignore:User-specified a background spectral grid is overridden by atmosphere spectral grid"
)
def test_rami4atm_gridded(mode_ckd_double, case, artefact_dir):
    specification = rami4atm.registry[case]
    ctor = specification.get("constructor")
    postprocess = specification.get("postprocess", lambda ls, _: ls[0])
    variables = specification.get("variables", ["radiance"])
    test_ctor = specification.get("test", SidakTTest)
    threshold = specification.get("threshold")

    srf_id, exps = ctor(spp=1000)

    resolution = (4, 3)
    resx, resy = resolution

    for exp in exps:
        mol_atm_1d = exp.atmosphere.molecular_atmosphere

        thermoprops_grid = [
            mol_atm_1d.thermoprops for _ in range(resx) for __ in range(resy)
        ]

        atm = GriddedHeterogeneousAtmosphere(
            molecular_atmosphere=GriddedMolecularAtmosphere(
                absorption_data=mol_atm_1d.absorption_data,
                thermoprops_grid=thermoprops_grid,
                grid_resolution=resolution,
                has_absorption=mol_atm_1d.has_absorption,
                has_scattering=mol_atm_1d.has_scattering,
            )
        )

        exp = attrs.evolve(exp, atmosphere=atm)

    srf = fresolver.load_dataset(f"srf/{srf_id}.nc")

    raw_results = [eradiate.run(exp) for exp in exps]

    result = postprocess(raw_results, srf)
    logger.info(result._repr_html_(), html=True)

    reference = fresolver.load_dataset(
        f"tests/regression_test_references/rami4atm/{case}-ref.nc"
    )
    logger.info(reference._repr_html_(), html=True)

    for variable in variables:
        test = test_ctor(
            name=case,
            value=result,
            reference=reference,
            threshold=threshold,
            archive_dir=artefact_dir,
            variable=variable,
            plot=False,
        )

        passed = test.run(diagnostic=True)
        assert passed
