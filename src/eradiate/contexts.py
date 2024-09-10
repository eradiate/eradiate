from __future__ import annotations

import typing as t

import attrs

from .attrs import define, documented
from .spectral import SpectralIndex

# ------------------------------------------------------------------------------
#                                      ABC
# ------------------------------------------------------------------------------


@attrs.define
class Context:
    """Abstract base class for all context data structures."""

    def evolve(self, **changes):
        """
        Create a copy of self with changes applied.

        Parameters
        ----------
        **changes
            Keyword changes in the new copy.

        Returns
        -------
        <same type as self>
            A copy of self with ``changes`` incorporated.
        """
        return attrs.evolve(self, **changes)


# ------------------------------------------------------------------------------
#                         Kernel dictionary contexts
# ------------------------------------------------------------------------------


@define
class KernelContext(Context):
    """
    Kernel evaluation context data structure. This class is used *e.g.* to store
    information about the spectral configuration to apply when generating kernel
    dictionaries or update parameter maps associated with a
    :class:`.SceneElement` instance.
    """

    si: SpectralIndex = documented(
        attrs.field(
            factory=SpectralIndex.new,
            converter=SpectralIndex.convert,
            validator=attrs.validators.instance_of(SpectralIndex),
        ),
        doc="Spectral index (used to evaluate quantities with any degree "
        "or kind of dependency vs spectrally varying quantities).",
        type=":class:`.SpectralIndex`",
        init_type=":class:`.SpectralIndex` or dict",
        default=":meth:`SpectralIndex.new() <.SpectralIndex.new>`",
    )

    kwargs: dict[str, t.Any] = documented(
        attrs.field(factory=dict),
        doc="Object-specific parameter overrides.",
        type="dict",
        default="{}",
    )

    @property
    def index_formatted(self) -> str:
        return self.si.formatted_repr
