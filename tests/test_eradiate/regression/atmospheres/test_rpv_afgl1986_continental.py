import os

import numpy as np
import pytest
import xarray as xr

import eradiate.scenes as esc
from eradiate.data import data_store
from eradiate.experiments import OneDimExperiment
from eradiate.test_tools.regression import Chi2Test
from eradiate.units import unit_registry as ureg


@pytest.mark.regression
def test_rpv_afgl1986_continental_brfpp(mode_ckd_double, metadata, session_timestamp):
    """
    This test case uses a basic atmospheric scene:

    - RPV surface emulating a canopy
    - Molecular atmosphere following the AFGL 1986 model
    - Aerosol layer at 1km elevation

    Aerosol layer parameters are:

    - bottom height: 1km
    - top height: 2km
    - n_layers: 16
    - tau_550: 0.5
    - radiative properties dataset: Govaerts 2021 Continental

    Simulation is run at 550nm with the RPV parameters chosen to match:

    - k: 0.95
    - g: -0.1
    - rho_0: 0.027685

    The remaining parameters are:

    - Sun zenith angle: 20°
    - Sun azimuth angle: 0°

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

    reference_path = data_store.fetch(
        "tests/regression_test_references/rpv_afgl1986_continental_brfpp_ref.nc"
    )
    reference = xr.load_dataset(reference_path)
    archive_path = metadata.get("archive_path", None)

    archive_filename = (
        os.path.join(
            archive_path,
            f"{session_timestamp:%Y%m%d-%H%M%S}-rpv_afgl1986_continental.nc",
        )
        if archive_path
        else None
    )

    test = Chi2Test(
        value=result,
        reference=reference,
        threshold=0.05,
        archive_filename=archive_filename,
    )

    assert test.run()
