import pytest

import eradiate
from eradiate import unit_registry as ureg


@pytest.mark.parametrize("geometry", ["plane_parallel", "spherical_shell"])
def test_heterogeneous_parameter_lookup(modes_all_double, geometry):

    atmosphere_kwargs = {
        "type": "molecular",
        "has_absorption": True,
        "has_scattering": False,
    }
    atmosphere_mono = {
        **atmosphere_kwargs,
        "construct": "ussa_1976",
        "absorption_dataset": "tests/spectra/absorption/us76_u86_4-spectra-4000_25711.nc",
    }
    atmosphere_ckd = {
        **atmosphere_kwargs,
        "construct": "afgl_1986",
    }

    exp = eradiate.experiments.AtmosphereExperiment(
        geometry=geometry,
        surface={"type": "lambertian", "reflectance": 1.0},
        atmosphere=atmosphere_ckd if eradiate.mode().is_ckd else atmosphere_mono,
        illumination={"type": "directional", "zenith": 30.0},
        measures={
            "type": "distant_flux",
            "id": "dflux",
            "srf": eradiate.scenes.spectra.MultiDeltaSpectrum(
                wavelengths=550.0 * ureg.nm
            ),
            "ray_offset": 1.0 * ureg.m,
        },
    )
    exp.init()

    # Medium is resolved, regardless the fact that it is first encountered as
    # a member of the "dflux" sensor
    assert (
        exp.mi_scene.umap_template["medium_atmosphere.albedo.volume.data"].parameter_id
        == "dflux.medium.albedo.volume.data"
        if geometry == "spherical_shell"
        else "dflux.medium.albedo.data"
    )
