"""
Data loading, conversion and output components.
"""

from __future__ import annotations

from .convert import libradtran_to_prt
from ..util.deprecation import deprecated


@deprecated(
    deprecated_in="1.3.0",
    removed_in="2.0.0",
    details="Use ``libradtran_to_prt()`` instead.",
)
def load_aerosol_libradtran(
    data,
    tolerance=None,
    wbounds=(None, None),
    fallback_units=None,
    **kwargs,
):
    """
    Compatibility alias for :func:`.libradtran_to_prt`.
    """
    kwargs.setdefault("squeeze", True)
    return libradtran_to_prt(
        data,
        fallback_units=fallback_units,
        tolerance=tolerance,
        wbounds=wbounds,
        **kwargs,
    )
