from __future__ import annotations

import typing as t

import attrs

from .attrs import documented, parse_docs
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


@parse_docs
@attrs.define
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


# ------------------------------------------------------------------------------
#                         Context Generator
# ------------------------------------------------------------------------------


class MultiGenerator:
    """
    This generator aggregates several generators and makes sure that items that
    have already been served are not repeated.
    """

    def __init__(self, generators):
        self.generators = generators
        self._i_generator = 0
        self._current_iterator = iter(self.generators[self._i_generator])
        self._visited = set()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            result = next(self._current_iterator)
            if result not in self._visited:
                self._visited.add(result)
                return result
            else:
                return self.__next__()
        except StopIteration:
            if self._i_generator >= len(self.generators) - 1:
                raise
            else:
                self._i_generator += 1
                self._current_iterator = iter(self.generators[self._i_generator])
                return self.__next__()
