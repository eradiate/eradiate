from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

import attr

from ..core import KernelDict, SceneElement
from ..._factory import Factory
from ...attrs import documented, get_doc, parse_docs
from ...contexts import KernelDictContext
from ...util.misc import onedict_value

surface_factory = Factory()
surface_factory.register_lazy_batch(
    [
        ("_basic.BasicSurface", "basic", {}),
        ("_central_patch.CentralPatchSurface", "central_patch", {}),
        ("_dem.DEMSurface", "dem", {})
    ],
    cls_prefix="eradiate.scenes.surface",
)


@parse_docs
@attr.s
class Surface(SceneElement, ABC):
    """
    An abstract base class defining common facilities for all surfaces.
    """

    id: t.Optional[str] = documented(
        attr.ib(
            default="surface",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        init_type=get_doc(SceneElement, "id", "init_type"),
        default='"surface"',
    )

    @property
    def shape_id(self) -> str:
        return f"shape_{self.id}"

    @property
    def bsdf_id(self) -> str:
        return f"bsdf_{self.id}"

    @abstractmethod
    def kernel_bsdfs(self, ctx: KernelDictContext) -> KernelDict:
        """
        Return BSDF plugin specifications.

        Parameters
        ----------
        ctx : :class:`.KernelDictContext`
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns
        -------
        :class:`.KernelDict`
            A kernel dictionary containing all the BSDFs attached to the
            surface.
        """
        pass

    @abstractmethod
    def kernel_shapes(self, ctx: KernelDictContext) -> KernelDict:
        """
        Return shape plugin specifications.

        Parameters
        ----------
        ctx : :class:`.KernelDictContext`
           A context data structure containing parameters relevant for kernel
           dictionary generation.

        Returns
        -------
        :class:`.KernelDict`
           A kernel dictionary containing all the shapes attached to the
           surface.
        """
        pass

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        # Inherit docstring
        kernel_dict = {
            self.bsdf_id: onedict_value(self.kernel_bsdfs(ctx)),
            self.shape_id: onedict_value(self.kernel_shapes(ctx)),
        }

        # This will overwrite any set BSDF
        kernel_dict[self.shape_id].update({"bsdf": {"type": "ref", "id": self.bsdf_id}})

        return KernelDict(kernel_dict)
