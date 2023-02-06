import typing as t
import warnings

import attrs

from ..core import CompositeSceneElement, traverse
from ...kernel import UpdateParameter


@attrs.define(eq=False, slots=False)
class GroupShape(CompositeSceneElement):
    shapes = attrs.field(kw_only=True)
    bsdfs = attrs.field(kw_only=True)
    instances = attrs.field(default=None)

    def update(self) -> None:
        # Normalisation strictly required to match Mitsuba and Eradiate IDs
        for objects in [self.shapes, self.bsdfs]:
            for key, value in objects.items():
                if value.id is not None and value.id != key:
                    warnings.warn(
                        f"While initializing GroupShape: ID '{value.id}' "
                        f"is normalized to '{key}'"
                    )
                value.id = key

    @property
    def template(self) -> dict:
        result = {}

        if not self.instances:
            for objects in [self.bsdfs, self.shapes]:
                for obj_key, obj in objects.items():
                    obj_template = traverse(obj)[0].data

                    for param_key, param_value in obj_template.items():
                        result[f"{obj_key}.{param_key}"] = param_value

        else:
            raise NotImplementedError

        return result

    @property
    def params(self) -> t.Dict[str, UpdateParameter]:
        result = {}

        if not self.instances:
            for objects in [self.bsdfs, self.shapes]:
                for obj_key, obj in objects.items():
                    obj_params = traverse(obj)[1].data

                    for param_key, param_value in obj_params.items():
                        result[f"{obj_key}.{param_key}"] = param_value

        else:
            raise NotImplementedError

        return result
