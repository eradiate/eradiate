"""Exceptions and warnings specific to Eradiate."""


# Exceptions
class ModeError(Exception):
    """Raised when encountering issues with Eradiate modes."""


class KernelVariantError(Exception):
    """Raised when encountering issues with Eradiate kernel variants."""
    pass


class UnitsError(Exception):
    """Raised when encountering issues with units (can be raised even when
    DimensionalityError would not)."""
    pass


# Warnings
class ConfigWarning(UserWarning):
    """Used when encountering nonfatal configuration issues."""
    pass
