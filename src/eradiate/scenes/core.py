from __future__ import annotations

import typing as t
import warnings
from abc import ABC, abstractmethod
from collections import abc as collections_abc
from typing import Mapping, Sequence

import attr
import mitsuba
import numpy as np
import pint
import pinttr
from pinttr.util import ensure_units

from .. import unit_context_config as ucc
from .._util import onedict_value
from ..attrs import documented, parse_docs
from ..contexts import KernelDictContext
from ..exceptions import KernelVariantError
from ..units import unit_registry as ureg


def _kernel_dict_get_mts_variant():
    variant = mitsuba.variant()

    if variant is not None:
        return variant
    else:
        raise KernelVariantError(
            "a kernel variant must be selected to create a KernelDict instance"
        )


@parse_docs
@attr.s
class KernelDict(collections_abc.MutableMapping):
    """
    A dictionary-like object designed to contain a scene specification
    appropriate for instantiation with :func:`~mitsuba.core.load_dict`.

    :class:`.KernelDict` keeps track of the variant it has been created with
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
        try:
            self.data.__getitem__(k)
            warnings.warn(
                f"Duplicate key '{k}' will be overwritten. Are you trying to "
                "add scene elements with duplicate IDs to this KernelDict?"
            )
        except KeyError:
            pass
        return self.data.__setitem__(k, v)

    def check(self) -> None:
        """
        Perform basic checks on the dictionary:

        * check that the ``"type"`` parameter is included;
        * check if the variant for which the kernel dictionary was created is
          the same as the current one.

        Raises
        ------
        ValueError
            If the ``"type"`` parameter is missing.

        :class:`.KernelVariantError`
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

    def fix(self) -> None:
        if "type" not in self.data:
            self.data["type"] = "scene"

    def load(
        self, strip: bool = True, post_load_update: bool = True
    ) -> mitsuba.core.Object:
        """
        Call :func:`~mitsuba.core.load_dict` on self. In addition, a
        post-load update can be applied.

        If the encapsulated dictionary misses a ``"type"`` key, it will be
        promoted to a scene dictionary through the addition of
        ``{"type": "scene"}``. For instance, it means that

        .. code:: python

           {
               "shape1": {"type": "sphere"},
               "shape2": {"type": "sphere"},
           }

        will be interpreted as

        .. code:: python

           {
               "type": "scene",
               "shape1": {"type": "sphere"},
               "shape2": {"type": "sphere"},
           }

        .. note::
           Requires a valid selected operational mode.

        Parameters
        ----------
        strip : bool
            If ``True``, if ``data`` has no ``'type'`` entry and if ``data``
            consists of one nested dictionary, it will be loaded directly.
            For instance, it means that

            .. code:: python

               {"phase": {"type": "rayleigh"}}

            will be stripped to

            .. code:: python

               {"type": "rayleigh"}

        post_load_update : bool
            If ``True``, use :func:`~mitsuba.python.util.traverse` and update
            loaded scene parameters according to data stored in ``post_load``.

        Returns
        -------
        :class:`mitsuba.core.Object`
            Loaded Mitsuba object.
        """
        from mitsuba.core import load_dict
        from mitsuba.python.util import traverse

        d = self.data
        d_extra = {}

        if "type" not in self:
            if len(self) == 1 and strip:
                # Extract plugin dictionary
                d = onedict_value(d)
            else:
                # Promote to scene dictionary
                d_extra = {"type": "scene"}

        obj = load_dict({**d, **d_extra})

        if self.post_load and post_load_update:
            params = traverse(obj)
            params.keep(list(self.post_load.keys()))
            for k, v in self.post_load.items():
                params[k] = v

            params.update()

        return obj

    def add(self, *elements: SceneElement, ctx: KernelDictContext) -> None:
        """
        Merge the content of a :class:`~eradiate.scenes.core.SceneElement` or
        another dictionary object with the current :class:`KernelDict`.

        Parameters
        ----------
        *elements : :class:`SceneElement`
            :class:`~eradiate.scenes.core.SceneElement` instances to add to the
            scene dictionary.

        ctx : :class:`.KernelDictContext`
            A context data structure containing parameters relevant for kernel
            dictionary generation. *This argument is keyword-only and required.*
        """
        for element in elements:
            self.update(element.kernel_dict(ctx))

    def merge(self, other: KernelDict) -> None:
        """
        Merge another :class:`.KernelDict` with the current one.

        Parameters
        ----------
        other : :class:`.KernelDict`
            A kernel dictionary whose main and post-load dictionaries will be
            used to update the current one.
        """
        if self.variant != other.variant:
            raise KernelVariantError("merged kernel dicts must share the same variant")

        self.data.update(other.data)
        self.post_load.update(other.post_load)

    @classmethod
    def from_elements(
        cls, *elements: SceneElement, ctx: KernelDictContext
    ) -> KernelDict:
        """
        Create a new :class:`.KernelDict` from one or more scene elements.

        Parameters
        ----------
        *elements : :class:`SceneElement`
            :class:`~eradiate.scenes.core.SceneElement` instances to add to the
            scene dictionary.

        ctx : :class:`.KernelDictContext`
            A context data structure containing parameters relevant for kernel
            dictionary generation. *This argument is keyword-only and required.*

        Returns
        -------
        :class:`KernelDict`
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

    id: t.Optional[str] = documented(
        attr.ib(
            default=None,
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc="User-defined object identifier.",
        type="str or None",
        init_type="str, optional",
        default="None",
    )

    def _kernel_dict_id(self) -> t.Dict:
        """
        Return a scene dictionary entry with the object's ``id`` field if it is
        not ``None``.
        """
        result = {}
        if self.id is not None:
            result["id"] = self.id
        return result

    @abstractmethod
    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        """
        Return a dictionary suitable for kernel scene configuration.

        Parameters
        ----------
        ctx : :class:`.KernelDictContext`
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns
        -------
        :class:`.KernelDict`
            Kernel dictionary which can be loaded as a Mitsuba object.
        """
        pass


@parse_docs
@attr.s(frozen=True)
class BoundingBox:
    """
    A basic data class representing an axis-aligned bounding box with
    unit-valued corners.

    Notes
    -----
    Instances are immutable.
    """

    min: pint.Quantity = documented(
        pinttr.ib(
            units=ucc.get("length"),
            on_setattr=None,  # frozen instance: on_setattr must be disabled
        ),
        type="quantity",
        init_type="array-like or quantity",
        doc="Min corner.",
    )

    max: pint.Quantity = documented(
        pinttr.ib(
            units=ucc.get("length"),
            on_setattr=None,  # frozen instance: on_setattr must be disabled
        ),
        type="quantity",
        init_type="array-like or quantity",
        doc="Max corner.",
    )

    @min.validator
    @max.validator
    def _min_max_validator(self, attribute, value):
        if not self.min.shape == self.max.shape:
            raise ValueError(
                f"while validating {attribute.name}: 'min' and 'max' must "
                f"have the same shape (got {self.min.shape} and {self.max.shape})"
            )
        if not np.all(np.less(self.min, self.max)):
            raise ValueError(
                f"while validating {attribute.name}: 'min' must be strictly "
                "less than 'max'"
            )

    @classmethod
    def convert(
        cls, value: t.Union[t.Sequence, t.Mapping, np.typing.ArrayLike, pint.Quantity]
    ) -> t.Any:
        """
        Attempt conversion of a value to a :class:`BoundingBox`.

        Parameters
        ----------
        value
            Value to convert.

        Returns
        -------
        any
            If `value` is an array-like, a quantity or a mapping, conversion will
            be attempted. Otherwise, `value` is returned unmodified.
        """
        if isinstance(value, (np.ndarray, pint.Quantity)):
            return cls(value[0, :], value[1, :])

        elif isinstance(value, Sequence):
            return cls(*value)

        elif isinstance(value, Mapping):
            return cls(**pinttr.interpret_units(value, ureg=ureg))

        else:
            return value

    @property
    def shape(self):
        """
        tuple: Shape of `min` and `max` arrays.
        """
        return self.min.shape

    @property
    def extents(self) -> pint.Quantity:
        """
        :class:`pint.Quantity`: Extent in all dimensions.
        """
        return self.max - self.min

    @property
    def units(self):
        """
        :class:`pint.Unit`: Units of `min` and `max` arrays.
        """
        return self.min.units

    def contains(self, p: np.typing.ArrayLike, strict: bool = False) -> bool:
        """
        Test whether a point lies within the bounding box.

        Parameters
        ----------
        p : quantity or array-like
            An array of shape (3,) (resp. (N, 3)) representing one (resp. N)
            points. If a unitless value is passed, it is interpreted as
            ``ucc["length"]``.

        strict : bool
            If ``True``, comparison is done using strict inequalities (<, >).

        Returns
        -------
        result : array of bool or bool
            ``True`` iff ``p`` in within the bounding box.
        """
        p = np.atleast_2d(ensure_units(p, ucc.get("length")))

        cmp = (
            np.logical_and(p > self.min, p < self.max)
            if strict
            else np.logical_and(p >= self.min, p <= self.max)
        )

        return np.all(cmp, axis=1)
