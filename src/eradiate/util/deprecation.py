from __future__ import annotations

import datetime
import functools
import inspect
import typing as t
import warnings

from packaging import version

from .numpydoc import format_doc, parse_doc
from .._version import _version

# Note: Most of this code is taken from https://github.com/briancurtin/deprecation.
# Class wrapping code is inspired by https://github.com/tantale/deprecated.


def deprecated(
    deprecated_in: str | None = None,
    removed_in: str | None = None,
    details: str = "",
    current_version: str = None,
):
    r"""
    Decorate a component to signify its deprecation

    This component wraps a component (function or class) that will soon be
    removed and does two things:

    * The docstring of the method will be modified to include a notice
      about deprecation, *e.g.* "Deprecated since 0.9.11. Use foo instead."
    * Raises a :class:`.DeprecatedWarning` via the :mod:`warnings` module,
      which is a subclass of the built-in :class:`DeprecationWarning`. Note that
      built-in :class:`DeprecationWarning`\ s are ignored by default, so for users
      to be informed of said warnings they will need to enable them---see
      the :mod:`warnings` module documentation for more details.

    Parameters
    ----------
    deprecated_in : str, optional, default: None
        The version at which the decorated method is considered deprecated.
        This will usually be the next version to be released when the decorator
        is added. The default is ``None``, which effectively means immediate
        deprecation. If this is not specified, then the `removed_in` and
        `current_version` arguments are ignored.

    removed_in : str, optional, default: None
        The version or :class:`datetime.date` when the decorated method will be
        removed. The default is ``None``, specifying that the component is not
        currently planned to be removed.
        Note: This parameter cannot be set to a value if `deprecated_in` is set
        to ``None``.

    details : str, optional
        Extra details to be added to the method docstring and warning. For
        example, the details may point users to a replacement method, such as
        "Use the foo_bar method instead". By default, there are no details.

    current_version : str, optional
        Current version. If unset, Eradiate's package version number is looked
        up automatically.

    Raises
    ------
    ValueError
        If `removed_in` is not ``None`` and `deprecated_in` is not ``None``.

    Examples
    --------
    This decorator allows for specifying the version from which the wrapped
    component is deprecated, and when it will be retired (this latter argument
    is optional):

    .. testsetup:: deprecated

       from eradiate.util.deprecation import deprecated

    .. doctest:: deprecated

       >>> @deprecated(deprecated_in="0.21.1", removed_in="0.22.1")
       ... def foo(): ...
       >>> foo()

    This will emit a :class:`.UnsupportedWarning`, which you can then handle.
    We can also deprecate classes:

    .. doctest:: deprecated

       >>> @deprecated(deprecated_in="0.21.1", removed_in="0.22.1")
       ... class SomeClass: ...

    If those components are used before v0.21.1, nothing happens. If they are
    used between v0.21.1 (included) and v0.22.1 (excluded), a
    :class:`.DeprecatedWarning` is emitted. If they are used from v0.22.1 on,
    an :class:`.UnsupportedWarning` is emitted.

    When used with static or class methods, their respective decorators must be
    applied *after* this one:

    .. doctest:: deprecated

       >>> class SomeClass:
       ...     @staticmethod
       ...     @deprecated(deprecated_in="0.21.1", removed_in="0.22.1")
       ...     def staticmethod(): ...
       ...
       ...     @classmethod
       ...     @deprecated(deprecated_in="0.21.1", removed_in="0.22.1")
       ...     def classmethod(cls): ...

    .. testcleanup:: deprecated

       del deprecated
    """
    # You can't just jump to removal. It's weird, unfair, and also makes
    # building up the docstring weird.
    if deprecated_in is None and removed_in is not None:
        raise ValueError(
            "Cannot set removed_in to a value without also setting deprecated_in"
        )

    # Only warn when it's appropriate. There may be cases when it makes sense
    # to add this decorator before a formal deprecation period begins.
    # In CPython, PendingDeprecatedWarning gets used in that period,
    # so perhaps mimic that at some point.
    is_deprecated = False
    is_unsupported = False

    # StrictVersion won't take a None or a "", so make whatever goes to it
    # is at least *something*. Compare versions only if removed_in is not
    # of type datetime.date
    if isinstance(removed_in, datetime.date):
        if datetime.date.today() >= removed_in:
            is_unsupported = True
        else:
            is_deprecated = True
    else:
        current_version = version.parse(
            _version if current_version is None else current_version
        )

        if removed_in is not None and current_version >= version.parse(removed_in):
            is_unsupported = True
        elif deprecated_in and current_version >= version.parse(deprecated_in):
            is_deprecated = True

    should_warn = any([is_deprecated, is_unsupported])

    def _wrapper(wrapped):
        if should_warn:
            # The various parts of this decorator being optional makes for
            # a number of ways the deprecation notice could go. The following
            # makes for a nicely constructed sentence with or without any
            # of the parts.

            # If removed_in is a date, use "removed on"
            # If removed_in is a version, use "removed in"
            parts = [".. deprecated::"]
            if deprecated_in:
                parts.append(f" {deprecated_in}")
            if removed_in:
                parts.append(
                    "\n   This will be removed "
                    f"{'on' if isinstance(removed_in, datetime.date) else 'in'} "
                    f"{removed_in}."
                )
            if details:
                parts.append(f"\n   {details}")

            deprecation_note = "".join(parts)

            # Parse and update wrapped object docstring
            doc = wrapped.__doc__ or ""  # Safety in case wrapped has no docstring
            sections = parse_doc(doc)
            sections["_deprecation"] = deprecation_note
            wrapped.__doc__ = "".join(format_doc(sections))

            # Define the actual emitted warning
            warning_cls = UnsupportedWarning if is_unsupported else DeprecatedWarning
            the_warning = warning_cls(
                wrapped.__name__, deprecated_in, removed_in, details
            )

        if inspect.isclass(
            wrapped
        ):  # We're wrapping a class: issue a warning upon class instantiation
            old_new = wrapped.__new__

            def wrapped_cls(cls, *args, **kwargs):
                if should_warn:
                    warnings.warn(
                        the_warning, category=DeprecationWarning, stacklevel=2
                    )

                if old_new is object.__new__:
                    return old_new(cls)

                else:  # actually, we don't know the real signature of *old_new*
                    return old_new(cls, *args, **kwargs)

            wrapped.__new__ = staticmethod(wrapped_cls)

            return wrapped

        else:  # We assume we're wrapping a function: issue a warning upon call

            @functools.wraps(wrapped)
            def _inner(*args, **kwargs):
                if should_warn:
                    warnings.warn(
                        the_warning, category=DeprecationWarning, stacklevel=2
                    )

                return wrapped(*args, **kwargs)

            return _inner

    return _wrapper


def substitute(subs: dict[str, tuple[type, dict[str, str]]]) -> t.Callable:
    """
    Generate a simple module :func:`__getattr__` which redirects outdated
    attribute lookups to current values with a deprecation warning.

    Parameters
    ----------
    subs: dict
        A dictionary with outdated names as keys. Values are 2-tuples consisting
        of the current substitute and a dictionary of keyword arguments passed
        to the :class:`DeprecatedWarning` constructor.

    Returns
    -------
    callable
        The generated module :func:`__getattr__` function.

    Example
    -------
    >>> __getattr__ = substitute(
    ...     {
    ...         "OneDimExperiment": (  # Old name
    ...             AtmosphereExperiment,  # New type
    ...             {"deprecated_in": "0.22.5", "removed_in": "0.22.7"},  # Keyword args
    ...         )
    ...     }
    ... )

    """

    def __getattr__(name):
        if name in subs:
            new_type, kwargs = subs[name]
            warnings.warn(
                DeprecatedWarning(
                    component=f"'{name}'",
                    details=f"Use '{new_type.__name__}' instead.",
                    **kwargs,
                ),
            )
            return new_type

        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    return __getattr__


class DeprecatedWarning(DeprecationWarning):
    """
    A warning class for deprecated methods

    This is a specialization of the built-in :class:`DeprecationWarning`,
    adding parameters that allow us to get information into the ``__str__``
    that ends up being sent through the :mod:`warnings` system.
    The attributes aren't able to be retrieved after the warning gets
    raised and passed through the system as only the class--not the
    instance--and message are what gets preserved.

    Parameters
    ----------
    component : type or callable
        The component being deprecated.

    deprecated_in : str, optional, default: None
        The version that `component` is deprecated in.

    removed_in : str or datetime, optional, default: None
        The version or :class:`datetime.date` specifying when `component` gets
        removed.

    details : str, optional
        Optional details about the deprecation. Most often this will include
        directions on what to use instead of the now deprecated code.
    """

    def __init__(self, component, deprecated_in, removed_in, details=""):
        # NOTE: The docstring only works for this class if it appears up
        # near the class name, not here inside __init__. I think it has
        # to do with being an exception class.
        self.component = component
        self.deprecated_in = deprecated_in
        self.removed_in = removed_in
        self.details = details
        super(DeprecatedWarning, self).__init__(
            component, deprecated_in, removed_in, details
        )

    def __str__(self):
        parts = [f"{self.component} is deprecated"]
        if self.deprecated_in:
            parts.append(f" as of {self.deprecated_in}")
        if self.removed_in:
            parts.append(
                " and will be removed "
                f"{'on' if isinstance(self.removed_in, datetime.date) else 'in'} "
                f"{self.removed_in}"
            )
        if any([self.deprecated_in, self.removed_in, self.details]):
            parts.append(".")
        if self.details:
            parts.append(f" {self.details}")

        return "".join(parts)


class UnsupportedWarning(DeprecatedWarning):
    """
    A warning class for methods to be removed

    This is a subclass of :class:`.DeprecatedWarning` and is used to output a
    proper message about a component being unsupported.
    """

    def __str__(self):
        parts = [f"{self.component} is unsupported as of {self.removed_in}."]
        if self.details:
            parts.append(f" {self.details}")

        return "".join(parts)
