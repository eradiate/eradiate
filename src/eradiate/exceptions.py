"""Exceptions and warnings specific to Eradiate."""

from pinttr.util import always_iterable

# ------------------------------------------------------------------------------
#                                   Exceptions
# ------------------------------------------------------------------------------


class ModeError(Exception):
    """Raised when encountering issues with Eradiate modes."""

    pass


class UnsetModeError(ModeError):
    """
    Raised when an operational mode is request but no active mode is selected.
    """

    def __str__(self):
        return (
            "No active mode selected. "
            "Select one by calling the `eradiate.set_mode()` function."
        )


class UnsupportedModeError(ModeError):
    """Raised when a feature is not supported for the currently selected mode."""

    def __init__(self, supported=None, unsupported=None, msg=None):
        super().__init__(msg)

        import eradiate

        self.mode = eradiate.mode().id if eradiate.mode() is not None else None
        self.supported = list(always_iterable(supported))
        self.unsupported = list(always_iterable(unsupported))

    def __str__(self):
        msg = self.args[0] if self.args and self.args[0] else "unsupported mode"
        details = [
            f"current mode: '{self.mode}'"
            if self.mode is not None
            else "no active mode"
        ]

        if self.supported:
            details.append(f"supported: {', '.join(self.supported)}")

        if self.unsupported:
            details.append(f"unsupported: {', '.join(self.unsupported)}")

        return f"{msg} ({'; '.join(details)})"


class DataError(Exception):
    """Raised when encountering issues with data."""

    pass


class TraversalError(Exception):
    """Raised when an error is encountered during scene element traversal."""

    pass


# ------------------------------------------------------------------------------
#                                   Warnings
# ------------------------------------------------------------------------------


class ConfigWarning(UserWarning):
    """Used when encountering nonfatal configuration issues."""

    pass


class OverriddenValueWarning(UserWarning):
    """Used when a user-defined value is overridden during execution."""

    pass
