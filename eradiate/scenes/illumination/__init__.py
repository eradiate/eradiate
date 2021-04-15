"""Illumination-related scene generation facilities.

.. admonition:: Registered factory members [:class:`IlluminationFactory`]
   :class: hint

   .. factorytable::
      :factory: IlluminationFactory
"""

from ._core import Illumination, IlluminationFactory
from ._constant import ConstantIllumination
from ._directional import DirectionalIllumination


