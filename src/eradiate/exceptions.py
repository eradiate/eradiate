"""Exceptions and warnings specific to Eradiate."""

from pinttr.util import always_iterable

import eradiate

# ------------------------------------------------------------------------------
#                                   Exceptions
# ------------------------------------------------------------------------------


class ModeError(Exception):
    """Raised when encountering issues with Eradiate modes."""

    pass


class UnsupportedModeError(ModeError):
    """Raised when a feature is not supported for the currently selected mode."""

    def __init__(self, supported=None, unsupported=None, msg=None):
        super(UnsupportedModeError, self).__init__(msg)
        self.mode = eradiate.mode().id if eradiate.mode() is not None else None
        self.supported = list(always_iterable(supported))
        self.unsupported = list(always_iterable(unsupported))

    def __str__(self):
        msg = f"'{self.mode}'" if self.mode is not None else "None"
        extra_msg = []

        if self.supported:
            extra_msg.append(f"supported: {', '.join(self.supported)}")

        if self.unsupported:
            extra_msg.append(f"unsupported: {', '.join(self.unsupported)}")

        if extra_msg:
            msg += f" ({'; '.join(extra_msg)})"

        return msg


class KernelVariantError(Exception):
    """Raised when encountering issues with Eradiate kernel variants."""

    pass


class DataError(Exception):
    """Raised when encountering issues with data."""

    pass


class TraversalError(Exception):
    """Raised when an error is encountered during scene element traversal."""


# ------------------------------------------------------------------------------
#                                   Warnings
# ------------------------------------------------------------------------------


class ConfigWarning(UserWarning):
    """Used when encountering nonfatal configuration issues."""

    pass


class OverriddenValueWarning(UserWarning):
    """Used when a user-defined value is overridden during execution."""

    pass
