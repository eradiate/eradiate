"""Spectrum-related scene generation facilities.

.. admonition:: Registered factory members [:class:`SpectrumFactory`]
   :class: hint

   .. factorytable::
      :factory: SpectrumFactory
"""

from ._core import Spectrum, SpectrumFactory
from ._solar_irradiance import SolarIrradianceSpectrum
from ._uniform import UniformSpectrum
