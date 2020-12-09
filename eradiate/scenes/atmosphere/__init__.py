"""Atmosphere-related scene generation facilities.

.. admonition:: Registered factory members [:class:`.AtmosphereFactory`]
    :class: hint

    .. factorytable::
       :factory: AtmosphereFactory
"""

from .base import AtmosphereFactory
from .heterogeneous import HeterogeneousAtmosphere
from .homogeneous import HomogeneousAtmosphere
