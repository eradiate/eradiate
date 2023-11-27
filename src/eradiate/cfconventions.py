from .units import symbol
from .units import unit_context_config as ucc

# Variable attributes, see section 3 of the CF conventions document
# https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#_description_of_the_data
ATTRIBUTES = {
    "radiation_wavelength": {
        "standard_name": "radiation_wavelength",
        "long_name": "wavelength",
        "units": symbol(ucc.get("wavelength")),
    },
    "radiation_wavenumber": {
        "standard_name": "radiation_wavenumber",
        "long_name": "wavelength",
        "units": symbol(ucc.get("wavenumber")),
    },
    "quantile": {  # @eradiate
        "standard_name": "quantile",
        "long_name": "quantile",
        "units": "1",
    },
}
