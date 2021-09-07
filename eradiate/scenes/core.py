from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import abc as collections_abc
from typing import MutableMapping, Optional

import attr
import mitsuba

from ..attrs import documented, parse_docs
from ..contexts import KernelDictContext
from ..exceptions import KernelVariantError


def _kernel_dict_get_mts_variant():
    variant = mitsuba.variant()

    if variant is not None:
        return variant
    else:
        raise KernelVariantError(
            "a kernel variant must be selected to create a KernelDict instance"
        )


@attr.s
class KernelDict(collections_abc.MutableMapping):
    """
    A dictionary-like object designed to contain a scene specification
    appropriate for instantiation with :func:`~mitsuba.core.xml.load_dict`.

    :class:`KernelDict` keeps track of the variant it has been created with
    and performs minimal checks to help prevent inconsistent scene creation.
    """

    data: dict = documented(
        attr.ib(
            factory=dict,
            converter=dict,
        ),
        doc="Scene dictionary.",
        default="{}",
        type="dict",
    )

    post_load: dict = documented(
        attr.ib(
            factory=dict,
            converter=dict,
        ),
        doc="Post-load update dictionary.",
        default="{}",
        type="dict",
    )

    variant: str = documented(
        attr.ib(
            factory=_kernel_dict_get_mts_variant,
            validator=attr.validators.instance_of(str),
        ),
        doc="Kernel variant for which the dictionary is created. Defaults to "
        "currently active variant (if any; otherwise raises).",
        type="str",
        default=":func:`mitsuba.set_variant`",
    )

    def __getitem__(self, k):
        return self.data.__getitem__(k)

    def __delitem__(self, v):
        return self.data.__delitem__(v)

    def __len__(self):
        return self.data.__len__()

    def __iter__(self):
        return self.data.__iter__()

    def __setitem__(self, k, v):
        return self.data.__setitem__(k, v)

    def check(self):
        """
        Perform basic checks on the dictionary:

        * check that the ``"type"`` parameter is included;
        * check if the variant for which the kernel dictionary was created is
          the same as the current one.

        Raises → ValueError
            If the ``"type"`` parameter is missing.

        Raises → :class:`.KernelVariantError`
            If the variant for which the kernel dictionary was created is
            not the same as the current one
        """
        variant = mitsuba.variant()
        if self.variant != variant:
            raise KernelVariantError(
                f"scene dictionary created for kernel variant '{self.variant}', "
                f"incompatible with current variant '{variant}'"
            )

        if "type" not in self:
            raise ValueError("kernel scene dictionary is missing a 'type' parameter")

    def load(self, post_load_update=True) -> mitsuba.core.Object:
        """
        Call :func:`~mitsuba.core.xml.load_dict` on self. In addition, a
        post-load update can be applied.

        .. note::
           Requires a valid selected operational mode.

        Parameter ``post_load_update`` (bool):
            If ``True``, use :func:`~mitsuba.python.util.traverse` and update
            loaded scene parameters according to data stored in ``post_load``.

        Return → :class:`mitsuba.core.Object`:
            Loaded Mitsuba object.
        """
        from mitsuba.core.xml import load_dict
        from mitsuba.python.util import traverse

        if "type" not in self:
            warnings.warn(
                "KernelDict is missing 'type' entry, adding type='scene'",
                UserWarning,
            )
            self["type"] = "scene"

        obj = load_dict(self.data)

        if self.post_load and post_load_update:
            params = traverse(obj)
            params.keep(list(self.post_load.keys()))
            for k, v in self.post_load.items():
                params[k] = v

            params.update()

        return obj

    def add(self, *elements: SceneElement, ctx: KernelDictContext):
        """
        Merge the content of a :class:`~eradiate.scenes.core.SceneElement` or
        another dictionary object with the current :class:`KernelDict`.

        Parameter ``*elements`` (:class:`SceneElement`):
            :class:`~eradiate.scenes.core.SceneElement` instances to add to the
            scene dictionary.

        Parameter ``ctx`` (:class:`.KernelDictContext`):
            A context data structure containing parameters relevant for kernel
            dictionary generation. *This argument is keyword-only and required.*
        """

        for element in elements:
            self.update(element.kernel_dict(ctx))

    def merge(self, other: KernelDict):
        """
        Merge another :class:`.KernelDict` with the current one.

        Parameter ``other`` (:class:`.KernelDict`):
            A kernel dictionary whose main and post-load dictionaries will be
            used to update the current one.
        """
        if self.variant != other.variant:
            raise KernelVariantError("merged kernel dicts must share the same variant")

        self.data.update(other.data)
        self.post_load.update(other.post_load)

    @classmethod
    def from_elements(cls, *elements: SceneElement, ctx: KernelDictContext):
        """
        Create a new :class:`.KernelDict` from one or more scene elements.

        Parameter ``*elements`` (:class:`SceneElement`):
            :class:`~eradiate.scenes.core.SceneElement` instances to add to the
            scene dictionary.

        Parameter ``ctx`` (:class:`.KernelDictContext`):
            A context data structure containing parameters relevant for kernel
            dictionary generation. *This argument is keyword-only and required.*

        Returns → :class:`KernelDict`:
            Created scene kernel dictionary.
        """
        result = cls()
        result.add(*elements, ctx=ctx)
        return result


@parse_docs
@attr.s
class SceneElement(ABC):
    """
    Abstract class for all scene elements.

    This abstract base class provides a basic template for all scene element
    classes. It is written using the `attrs <https://www.attrs.org>`_ library.
    """

    id: Optional[str] = documented(
        attr.ib(
            default=None,
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc="User-defined object identifier.",
        type="str or None",
        default="None",
    )

    @abstractmethod
    def kernel_dict(self, ctx: KernelDictContext) -> MutableMapping:
        """
        Return a dictionary suitable for kernel scene configuration.

        Parameter ``ref`` (bool):
            If ``True``, use referencing for all relevant nested kernel plugins.

        Parameter ``ctx`` (:class:`.KernelDictContext`):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → dict:
            Dictionary suitable for merge with a kernel scene dictionary
            (using :func:`~mitsuba.core.xml.load_dict`).
        """
        pass
