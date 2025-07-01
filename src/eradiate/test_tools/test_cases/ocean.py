import joseki
import numpy as np

from ... import fresolver
from ... import unit_registry as ureg
from ...experiments import AtmosphereExperiment

WAVELENGTH = [412, 443, 550, 670, 865, 1020, 1600, 2200]
ETA = [1.349303, 1.346833, 1.341266, 1.337636, 1.336949, 1.336949, 1.336949, 1.336949]
WB_COASTAL = [
    3.4678e-02,
    4.1939e-02,
    6.0228e-02,
    5.7141e-02,
    0.0000e00,
    0.0000e00,
    0.0000e00,
    0.0000e00,
]
WB_OPEN = [
    6.7215e-02,
    6.5480e-02,
    4.4756e-02,
    1.7900e-02,
    0.0000e00,
    0.0000e00,
    0.0000e00,
    0.0000e00,
]


def ocean_grasp_wavelength():
    return WAVELENGTH


def create_ocean_grasp(water_body_reflectance, wind_speed, has_atmoshphere=False):
    """
    Create the reference scene of the 3DREAMS project with ocean surface.

    Parameters
    ----------
    water_body_reflectance : list {WB_COASTAL, WB_OPEN}
        The water body reflectance spectrum. Should either be the WB_COASTAL or
        WB_OPEN initialized above.

    wind_speed: float
        Wind speed at mast height in m/s.

    has_atmosphere: bool
        Flags whether to include an atmosphere and particle layer as described
        by 3DREAMS scenario REF_OO_UB01_I_S20_PPL.

    Returns
    -------
    AtmosphereExperiment
        The atmosphere experiment corresponding to the GRASP scenario.
    """

    if has_atmoshphere:
        UB = fresolver.load_dataset(
            "tests/regression_test_specifications/ocean_grasp/REF_UB.nc"
        )

        spp = 10000
        atmosphere = {
            "type": "heterogeneous",
            "molecular_atmosphere": {
                "type": "molecular",
                "has_absorption": False,
                "thermoprops": joseki.make(
                    identifier="afgl_1986-us_standard",
                    z=np.arange(0, 40 + 0.1, 0.1) * ureg.km,
                ).joseki.rescale_to({"CO2": 360 * ureg.ppm}),
                "rayleigh_depolarization": 0.0,
            },
            "particle_layers": [
                {
                    "type": "particle_layer",
                    "bottom": 0 * ureg.km,
                    "top": 40 * ureg.km,
                    "distribution": {
                        "type": "exponential",
                        "rate": 40,
                    },
                    "tau_ref": 0.1,
                    "w_ref": 550 * ureg.nm,
                    "has_absorption": True,
                    "dataset": UB,
                }
            ],
        }
    else:
        spp = 1
        atmosphere = None

    config = {
        "surface": {
            "type": "ocean_grasp",
            "wind_speed": wind_speed,
            "eta": {
                "type": "interpolated",
                "wavelengths": WAVELENGTH,
                "values": ETA,
            },
            "water_body_reflectance": {
                "type": "interpolated",
                "wavelengths": WAVELENGTH,
                "values": water_body_reflectance,
            },
        },
        "illumination": {
            "zenith": 20.0,
            "zenith_units": "degree",
            "azimuth": 0.0,
            "azimuth_units": "degree",
            "type": "directional",
        },
        "measures": [
            {
                "type": "mdistant",
                "construct": "hplane",
                "zeniths": np.arange(-60, 61, 5),
                "zeniths_units": "degree",
                "azimuth": 0.0,
                "azimuth_units": "degree",
                "srf": {
                    "type": "multi_delta",
                    "wavelengths": WAVELENGTH,
                },
                "spp": spp,
            }
        ],
        "geometry": {
            "type": "plane_parallel",
            "zgrid": np.arange(0, 40 + 0.1, 0.1) * ureg.km,
            "toa_altitude": 40 * ureg.km,
        },
        "atmosphere": atmosphere,
        "integrator": {"type": "piecewise_volpath", "moment": True}
        if has_atmoshphere
        else {"type": "volpath", "moment": True},
    }

    return AtmosphereExperiment(**config)


def create_ocean_grasp_coastal_no_atm():
    """
    Ocean GRASP Coastal with no atmosphere
    ======================================

    This test case includes a coastal oceanic surface with no atmosphere.

    Parameters
    ----------

    * Surface: Square surface with Ocean GRASP BSDF with wind speed of 2 m/s and
    an coastal ocean water body reflectance spectrum.
    * Illumination: Directional illumination with a zenith angle :math:`\theta = 20°`
    * Sensor: Multi Distant measure over the principal plane from -60 to 60 degrees
    with 5 degree steps.
    """
    return create_ocean_grasp(WB_COASTAL, 2.0, False)


def create_ocean_grasp_open_no_atm():
    """
    Ocean GRASP Open with no atmosphere
    ===================================

    This test case includes an open oceanic surface with no atmosphere.

    Parameters
    ----------

    * Surface: Square surface with Ocean GRASP BSDF with wind speed of 10 m/s and
    an open ocean water body reflectance spectrum.
    * Illumination: Directional illumination with a zenith angle :math:`\theta = 20°`
    * Sensor: Multi Distant measure over the principal plane from -60 to 60 degrees
    with 5 degree steps.
    """
    return create_ocean_grasp(WB_OPEN, 10.0, False)


def create_ocean_grasp_open_atm():
    """
    Ocean GRASP Open with atmosphere
    ===================================

    This test case includes an open oceanic surface with atmosphere.

    Parameters
    ----------

    * Surface: Square surface with Ocean GRASP BSDF with wind speed of 10 m/s and
    an open ocean water body reflectance spectrum.
    * Atmosphere (if `has_atmosphere=True`): afgl US standard atmosphere with
    rescaled CO2 concentration to 360 ppm and depolarization set to zero.
    * Particle Layer (if `has_atmosphere=True): Exponentially distributed layer
    with absorption and optical depth of 0.1, distribution
    * Illumination: Directional illumination with a zenith angle :math:`\theta = 20°`
    * Sensor: Multi Distant measure over the principal plane from -60 to 60 degrees
    with 5 degree steps.
    """
    return create_ocean_grasp(WB_OPEN, 10.0, True)
