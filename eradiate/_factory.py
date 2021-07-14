from typing import Any, MutableMapping, Optional, Tuple, Type, Union

import attr
import dessinemoi
import pinttr

from .units import unit_registry as ureg


@attr.s
class Factory(dessinemoi.Factory):
    """
    This factory subclass extends the original :class:`dessinemoi.Factory` type
    by adding the following functionality to its :meth:`.convert` method:

    * unit interpretation using :func:`pinttr.interpret_units`;
    * class method constructor selection using the ``construct`` key (overrides
      the default dict constructor if any).
    """

    def _convert_impl(
        self,
        value,
        allowed_cls: Optional[Union[Type, Tuple[Type]]] = None,
    ) -> Any:
        if isinstance(value, MutableMapping):
            # Interpret units and copy value to avoid unintended mutation
            value_copy = pinttr.interpret_units(value, ureg=ureg)

            # Query registry
            type_id = value_copy.pop("type")
            entry = self.registry[type_id]

            # Check if class is allowed
            if allowed_cls is not None and not issubclass(entry.cls, allowed_cls):
                raise TypeError(
                    f"conversion to object type '{type_id}' ({entry.cls}) is not allowed"
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
