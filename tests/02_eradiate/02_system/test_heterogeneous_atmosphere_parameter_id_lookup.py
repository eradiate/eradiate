import pytest

import eradiate
from eradiate import unit_registry as ureg


@pytest.mark.parametrize("geometry", ["plane_parallel", "spherical_shell"])
def test_heterogeneous_parameter_lookup(modes_all_double, geometry):
    exp = eradiate.experiments.AtmosphereExperiment(
        geometry=geometry,
        surface={"type": "lambertian", "reflectance": 1.0},
        atmosphere={
            "type": "molecular",
            "has_absorption": True,
            "has_scattering": False,
            "construct": "afgl_1986" if eradiate.mode().is_ckd else "ussa_1976",
        },
        illumination={"type": "directional", "zenith": 30.0},
        measures={
            "type": "distant_flux",
            "id": "dflux",
            "spectral_cfg": {"bins": "550"}
            if eradiate.mode().is_ckd
            else {"wavelengths": [550.0]},
            "ray_offset": 1.0 * ureg.m,
        },
    )
    exp.init()

    # Medium is resolved, regardless the fact that it is first encountered as
    # a member of the "dflux" sensor
    assert (
        exp.mi_scene.umap_template["medium_atmosphere.albedo.data"].parameter_id
        == "dflux.medium.albedo.volume.data"
        if geometry == "spherical_shell"
        else "dflux.medium.albedo.data"
    )
