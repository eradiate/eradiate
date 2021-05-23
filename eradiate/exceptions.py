"""Exceptions and warnings specific to Eradiate."""


from pinttr.util import always_iterable

import eradiate

# -- Exceptions ----------------------------------------------------------------


class ModeError(Exception):
    """Raised when encountering issues with Eradiate modes."""

    pass


class UnsupportedModeError(ModeError):
    """Raised when a feature is not supported for the currently selected mode."""

    def __init__(self, supported=None, unsupported=None, msg=None):
        super(UnsupportedModeError, self).__init__(msg)
        self.mode = eradiate.mode().id
        self.supported = list(always_iterable(supported))
        self.unsupported = list(always_iterable(unsupported))

    def __str__(self):
        msg = f"'{eradiate.mode().id}'"

        if self.supported:
            supported_extra_msg = f"supported: {', '.join(self.supported)}"
        else:
            supported_extra_msg = ""

        if self.unsupported:
            unsupported_extra_msg = f"unsupported: {', '.join(self.unsupported)}"
        else:
            unsupported_extra_msg = ""

        if supported_extra_msg or unsupported_extra_msg:
            msg += f" ({'; '.join((supported_extra_msg, unsupported_extra_msg))})"

        return msg


class KernelVariantError(Exception):
    """Raised when encountering issues with Eradiate kernel variants."""

    pass


# -- Warnings ------------------------------------------------------------------


class ConfigWarning(UserWarning):
    """Used when encountering nonfatal configuration issues."""

    pass


class OverriddenValueWarning(UserWarning):
    """Used when a user-defined value is overridden during execution."""

    pass
