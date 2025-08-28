"""
Extensions and helpers for tutorials (legacy).
"""

import warnings

from eradiate.tutorials import *  # noqa: F403

warnings.warn(
    DeprecationWarning(
        "The 'eradiate.notebook.tutorials' module is deprecated: its features "
        "have been moved to 'eradiate.tutorials'."
    )
)
