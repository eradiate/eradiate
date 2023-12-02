from __future__ import annotations

import typing as t
from collections.abc import MutableMapping

import attrs
import dessinemoi
import pinttr

from .units import unit_registry as ureg


@attrs.define
class Factory(dessinemoi.Factory):
    """
    Object factory.

    This factory subclass extends the original :class:`dessinemoi.Factory` type
    by adding the following functionality to its :meth:`convert` method:

    * unit interpretation using :func:`pinttr.interpret_units`;
    * class method constructor selection using the ``"construct"`` key
      (overrides the default dict constructor if any; reserves the
      ``"construct"`` parameter).

    The interface is unchanged.
    """

    def _convert_impl(
        self,
        value,
        allowed_cls: type | tuple[type] | None = None,
    ) -> t.Any:
        if isinstance(value, MutableMapping):
            # Interpret units and copy value to avoid unintended mutation
            value_copy = pinttr.interpret_units(value, ureg=ureg)

            # Query registry
            type_id = value_copy.pop("type")

            try:
                entry = self.registry[type_id]
            except KeyError as e:
                raise ValueError(f"no type registered as '{type_id}'") from e

            # Resolve lazy type if necessary
            cls = (
                entry.cls.load()
                if isinstance(entry.cls, dessinemoi.LazyType)
                else entry.cls
            )

            # Check if class is allowed
            if allowed_cls is not None and not issubclass(cls, allowed_cls):
                raise TypeError(
                    f"conversion to object type '{type_id}' ({cls}) is not allowed"
                )

            # Get constructor from dict, if any
            dict_constructor = value_copy.pop("construct", entry.dict_constructor)

            # Construct object
            return self.create(type_id, construct=dict_constructor, kwargs=value_copy)

        else:
            # Check if object has allowed type
            if allowed_cls is not None:
                if not isinstance(value, allowed_cls):
                    raise TypeError("value type is not allowed")

            return value

    def register_lazy_batch(
        self,
        specs: list[tuple[str, str, dict]],
        cls_prefix: str = "",
    ):
        """
        Register multiple lazy types at once.

        Parameters
        ----------
        specs : list of tuple[str, str, dict]
            A list of ``(cls, type_id, kwargs)`` tuples where ``cls`` is  the
            name of the target type (relative to `cls_prefix`), ``type_id`` is
            the ID for the registered type and ``kwargs`` is a dictionary
            containing keyword arguments for the :meth:`.Factory.register`
            method.

        cls_prefix : str
            A prefix relative to which lazy type names are expressed.
        """
        for cls, type_id, kwargs in specs:
            if cls_prefix:
                cls = f"{cls_prefix}.{cls}"
            self.register(cls, type_id=type_id, **kwargs)
