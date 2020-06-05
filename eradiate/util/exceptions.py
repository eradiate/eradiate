"""Exceptions and warnings specific to Eradiate."""


# Exceptions
class KernelVariantError(Exception):
    """Raised when encountering issues with Eradiate kernel variants."""
    pass


# Warnings
class ConfigWarning(UserWarning):
    """Used when encountering nonfatal configuration issues."""
    pass
