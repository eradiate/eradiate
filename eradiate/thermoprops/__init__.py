"""
Atmospheric thermophysical properties.
"""
from eradiate.xarray.metadata import DatasetSpec, VarSpec

profile_dataset_spec = DatasetSpec(
    var_specs={
        "p": VarSpec(standard_name="air_pressure", units="Pa", long_name="air pressure"),
        "t": VarSpec(standard_name="air_temperature", units="K", long_name="air temperature"),
        "n": VarSpec(standard_name="air_number_density", units="m^-3", long_name="air number density"),
        "mr": VarSpec(standard_name="mixing_ratio", units="", long_name="mixing ratio"),
    },
    coord_specs="atmospheric_profile"
)
