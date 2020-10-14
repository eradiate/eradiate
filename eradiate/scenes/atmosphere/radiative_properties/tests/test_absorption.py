import numpy as np

from eradiate.util.units import ureg
from eradiate.scenes.atmosphere.radiative_properties.absorption import sigma_a


def test_sigma_a():
    """Test function with default value
    """
    x = sigma_a()

    # test that the function returns a quantity that holds an array of values
    assert isinstance(x, ureg.Quantity)
    assert isinstance(x.magnitude, np.ndarray)

    # test that units are m^-1
    assert x.units == ureg.Unit("m^-1")
