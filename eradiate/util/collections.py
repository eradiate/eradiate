"""Specialised container datatypes providing alternatives to Python’s general
purpose built-in containers, dict, list, set, and tuple."""

from collections.abc import Sequence
from numbers import Number

import numpy
import pint
from dpath import util as dpu
from dpath.exceptions import PathNotFound


def is_vector3(value):
    """Returns ``True`` if a value can be interpreted as a 3-vector."""

    if isinstance(value, pint.Quantity):
        return is_vector3(value.magnitude)

    # @formatter:off
    return (
        (
            isinstance(value, numpy.ndarray) or
            (isinstance(value, Sequence) and not isinstance(value, str))
        )
        and len(value) == 3
        and all(map(lambda x: isinstance(x, Number), value))
    )
    # @formatter:on


def onedict_value(d):
    """Get the value of a single-entry dictionary."""

    if len(d) != 1:
        raise ValueError(f"dictionary has wrong length (expected 1, got {len(d)}")

    return next(iter(d.values()))


class ndict(dict):
    """A nested dict structure. Keys are expected to be strings (untested with
    different key types).
    """

    # Requires dpath [https://github.com/akesterson/dpath-python]

    def __init__(self, d={}, separator="."):
        """Initialise from another dictionary.

        Parameter ``d`` (dict)
            Dictionary to initialise from.

        Parameter ``separator`` (str)
            Key separator.

        """
        super().__init__(d)
        self.separator = separator

    def update(self, other):
        """Recursively update with content of another nested dict structure.
        Existing leaves are overwritten.

        Parameter ``other`` (dict):
            Dictionary to update ``self`` with.
        """
        dpu.merge(self, other, separator=self.separator, flags=dpu.MERGE_REPLACE)

    def rget(self, key):
        """Recursively access an element in the nested dictionary.

        Parameter ``key`` (str)
            Path to the queried element. The path separator is defined by
            ``self.separator``.

        Returns → object
            Requested object.

        Raises → ``KeyError``
            The requested key could not be found.
        """
        return dpu.get(self, key, separator=self.separator)

    def rset(self, key, value):
        """Set an element in the nested dictionary.

        Parameter ``key`` (str)
            Path to the element to set. The path separator is defined by
            ``self.separator``.

        Raises → ``KeyError``
            The requested key could not be found.
        """
        try:
            dpu.new(self, key, value, separator=self.separator)
        except PathNotFound:
            raise KeyError(key)
