"""Atmosphere-related scene generation facilities.

.. admonition:: Registered factory members [:class:`.AtmosphereFactory`]
    :class: hint

    .. factorytable::
       :factory: AtmosphereFactory
"""

from ._base import (
    Atmosphere,
    AtmosphereFactory
)
from ._heterogeneous import HeterogeneousAtmosphere
from ._homogeneous import HomogeneousAtmosphere


__all__ = [
    "Atmosphere",
    "AtmosphereFactory",
    "HeterogeneousAtmosphere",
    "HomogeneousAtmosphere",
]
