import numpy as np
import pytest

import eradiate.scenes as esc
from eradiate.data import data_store
from eradiate.experiments import OneDimExperiment
from eradiate.test_tools.regression import Chi2Test
from eradiate.units import unit_registry as ureg


@pytest.mark.regression
def test_rpv_afgl1986_continental_brfpp(
    mode_ckd_double, artefact_dir, session_timestamp
):
    r"""
    RPV AFGL1986 Aerosol regression test
    ====================================

    This is a regression test, which compares the simulation results of the
    current branch to an older reference version.

    Rationale
    ---------

    This test case uses a basic atmospheric scene:

    * RPV surface emulating a canopy
    * Molecular atmosphere following the AFGL 1986 model
    * Aerosol layer at 1km elevation

    Parameters

    * Atmosphere: Molecular atmosphere using the agfl1986 profile
    * Aerosol layer: 16 layers from 1km to 2km height, :math:`\tau_{500} = 0.5`
      Radiative properties from the Govaerts 2021 Continental dataset
    * Surface: Square surface with RPV BSDF with :math:`k = 0.95`, :math:`g = -0.1`
      and :math:`\rho_0 = 0.027685`
    * Illumination: Directional illumination with a zenith angle :math:`\theta = 20°`
    * Sensor: Distant reflectance measure, covering a plane, (76 angular points,
      10000 samples per pixel)

    Expected behaviour
    ------------------

    This test uses the Chi-squared criterion with a threshold of 0.05.

    """
    particle_layer = esc.atmosphere.ParticleLayer(
        bottom=1 * ureg.km,
        top=2 * ureg.km,
        n_layers=16,
        tau_550=0.5,
        dataset=data_store.fetch("spectra/particles/govaerts_2021-continental.nc"),
    )

    exp = OneDimExperiment(
        surface=esc.bsdfs.RPVBSDF(k=0.95, g=-0.1, rho_0=0.027685),
        illumination=esc.illumination.DirectionalIllumination(
            zenith=20 * ureg.deg, irradiance=20.0
        ),
        measures=[
            esc.measure.MultiDistantMeasure.from_viewing_angles(
                azimuths=0.0,
                zeniths=np.arange(-75.0, 75.01, 2.0),
                spp=10000,
                spectral_cfg=esc.measure.MeasureSpectralConfig.new(
                    bin_set="10nm", bins=["550"]
                ),
            )
        ],
        atmosphere=esc.atmosphere.HeterogeneousAtmosphere(
            molecular_atmosphere=esc.atmosphere.MolecularAtmosphere.afgl_1986(),
            particle_layers=[particle_layer],
        ),
    )

    exp.run()
    result = exp.results["measure"]

    test = Chi2Test(
        name=f"{session_timestamp:%Y%m%d-%H%M%S}-rpv_afgl1986_continental.nc",
        value=result,
        reference="tests/regression_test_references/rpv_afgl1986_continental_brfpp_ref.nc",
        threshold=0.05,
        archive_dir=artefact_dir,
    )

    assert test.run()
