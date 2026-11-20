"""
Deprecated heterogeneous atmosphere alias.
"""

from ._atmosphere import Atmosphere
from ...util.deprecation import substitute

__getattr__ = substitute(
    {
        "AbstractHeterogeneousAtmosphere": (
            Atmosphere,
            {"deprecated_in": "1.4.0", "removed_in": "2.0.0"},
        ),
        "HeterogeneousAtmosphere": (
            Atmosphere,
            {
                "deprecated_in": "1.4.0",
                "removed_in": "2.0.0",
                "details": "Use 'Atmosphere' (type ID 'atmosphere') instead.",
            },
        ),
    },
    stacklevel=3,
)
