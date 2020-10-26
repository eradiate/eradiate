"""Atmosphere-related scene generation facilities.

.. admonition:: Registered factory members
    :class: hint

    .. factorytable::
       :factory: SceneElementFactory
       :modules: eradiate.scenes.atmosphere
"""

from .heterogeneous import HeterogeneousAtmosphere
from .homogeneous import RayleighHomogeneousAtmosphere
