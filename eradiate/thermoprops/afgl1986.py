"""
Atmospheric thermophysical properties profiles models according to
:cite:`Anderson1986AtmosphericConstituentProfiles`.

These atmospheric profiles may be referred to as the AFGL 1986 atmospheric
profiles in other parts of the documentation.
"""
import eradiate.data as data


def make_profile(model_id="us_standard"):
    """Makes the atmospheric profiles from the AFGL's 1986 technical report
    :cite:`Anderson1986AtmosphericConstituentProfiles`.

    Parameter ``model_id`` (str):
        Choose from ``"midlatitude_summer"``, ``"midlatitude_winter"``,
        ``"subarctic_summer"``, ``"subarctic_winter"``, ``"tropical"`` and
        ``"us_standard"``.

        Default: ``"us_standard"``

    Returns → :class:`xarray.Dataset`:
        Atmospheric profile.
    """
    return data.open(category="thermoprops_profiles", id="afgl1986-" + model_id)
