import os
from abc import ABC, abstractmethod
from typing import MutableMapping, Optional

import attr
import numpy as np
import pinttr

from ..core import SceneElement
from ... import unit_context_kernel as uck
from ... import unit_registry as ureg
from ... import validators
from ..._factory import Factory
from ...attrs import documented, get_doc, parse_docs
from ...contexts import KernelDictContext
from ...units import unit_context_config as ucc

biosphere_factory = Factory()


@parse_docs
@attr.s
class Canopy(SceneElement, ABC):
    """
    An abstract base class defining a base type for all canopies.
    """

    id = documented(
        attr.ib(
            default="canopy",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        default='"canopy"',
    )

    size = documented(
        pinttr.ib(
            default=None,
            validator=attr.validators.optional(
                [
                    pinttr.validators.has_compatible_units,
                    validators.on_quantity(validators.is_vector3),
                ]
            ),
            units=ucc.deferred("length"),
        ),
        doc="Canopy size as a 3-vector.\n\nUnit-enabled field (default: ucc[length]).",
        type="array-like",
    )

    @abstractmethod
    def bsdfs(self, ctx=None):
        """
        Return BSDF plugin specifications only.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → dict:
            Return a dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the BSDFs
            attached to the shapes in the canopy.
        """
        pass

    @abstractmethod
    def shapes(self, ctx=None):
        """
        Return shape plugin specifications only.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → dict:
            A dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the shapes
            in the canopy.
        """
        pass

    @abstractmethod
    def kernel_dict(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
        """
        Return a dictionary suitable for kernel scene configuration.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → dict:
            Dictionary suitable for merge with a kernel scene dictionary
            (using :func:`~mitsuba.core.xml.load_dict`).
        """
        pass


@parse_docs
@attr.s
class CanopyElement(SceneElement, ABC):
    """
    An abstract class representing a component of a :class:`.Canopy` object.
    Concrete canopy classes can manage their components as they prefer.
    """

    @abstractmethod
    def bsdfs(self, ctx=None):
        """
        Return BSDF plugin specifications only.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → dict:
            Return a dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the BSDFs
            attached to the shapes in the canopy.
        """
        pass

    @abstractmethod
    def shapes(self, ctx=None):
        """
        Return shape plugin specifications only.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → dict:
            A dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the shapes
            in the canopy.
        """
        pass

    def kernel_dict(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
        if not ctx.ref:
            return self.shapes(ctx=ctx)
        else:
            return {**self.bsdfs(ctx=ctx), **self.shapes(ctx=ctx)}


@parse_docs
@attr.s
class InstancedCanopyElement(SceneElement):
    """
    Specification a canopy element, alongside the locations of instances (*i.e.*
    clones) of it.

    .. admonition:: Class method constructors

       .. autosummary::

          convert
          from_dict
          from_file
    """

    canopy_element = documented(
        attr.ib(
            default=None,
            validator=attr.validators.optional(
                attr.validators.instance_of(CanopyElement)
            ),
            converter=biosphere_factory.convert,
        ),
        doc="Instanced canopy element. Can be specified as a dictionary, which will "
        "be interpreted by :meth:`.CanopyElement.from_dict`.",
        type=":class:`.CanopyElement`",
        default="None",
    )

    instance_positions = documented(
        pinttr.ib(factory=list, units=ucc.deferred("length")),
        doc="Instance positions as an (n, 3)-array.\n"
        "\n"
        "Unit-enabled field (default: ucc[length])",
        type="array-like",
        default="[]",
    )

    @instance_positions.validator
    def _instance_positions_validator(self, attribute, value):
        if value.shape and value.shape[0] > 0 and value.shape[1] != 3:
            raise ValueError(
                f"while validating {attribute.name}, must be an array of shape "
                f"(n, 3), got {value.shape}"
            )

    # --------------------------------------------------------------------------
    #                               Constructors
    # --------------------------------------------------------------------------

    @classmethod
    def from_file(cls, filename=None, canopy_element=None):
        """
        Construct a :class:`.InstancedCanopyElement` from a text file specifying
        instance positions.

        .. admonition:: File format

           Each line defines an instance position as a whitespace-separated
           3-vector of Cartesian coordinates.

        .. important::

           Location coordinates are assumed to be given in meters.

        Parameter ``filename`` (path-like):
            Path to the text file specifying the leaves in the canopy.
            Can be absolute or relative. Required (setting to ``None`` will
            raise an exception).

        Parameter ``canopy_element`` (:class:`.CanopyElement` or dict):
            :class:`.CanopyElement` to be instanced. If a dictionary is passed,
            if is interpreted by :meth:`.CanopyElement.from_dict`. If set to
            ``None``, an empty leaf cloud will be created.

        Returns → :class:`.InstancedCanopyElement`:
            Created :class:`.InstancedCanopyElement`.

        Raises → ValueError:
            If ``filename`` is set to ``None``.

        Raises → FileNotFoundError:
            If ``filename`` does not point to an existing file.
        """
        if filename is None:
            raise ValueError("parameter 'filename' is required")

        if not os.path.isfile(filename):
            raise FileNotFoundError(f"no file at {filename} found.")

        if canopy_element is None:
            canopy_element = {"type": "leaf_cloud"}

        canopy_element = biosphere_factory.convert(canopy_element)

        instance_positions = []

        with open(filename, "r") as f:
            for i_line, line in enumerate(f):
                try:
                    coords = np.array(line.split(), dtype=float)
                except ValueError as e:
                    raise ValueError(
                        f"while reading {filename}, on line {i_line + 1}: "
                        f"cannot convert {line} to a 3-vector!"
                    ) from e

                if len(coords) != 3:
                    raise ValueError(
                        f"while reading {filename}, on line {i_line + 1}: "
                        f"cannot convert {line} to a 3-vector!"
                    )

                instance_positions.append(coords)

        instance_positions = np.array(instance_positions) * ureg.m
        return cls(canopy_element=canopy_element, instance_positions=instance_positions)

    @classmethod
    def from_dict(cls, d):
        """
        Construct from a dictionary. This function first queries for a
        ``type`` parameter. If it is found, dictionary parameters are used to
        call another class method constructor:

        * ``file``: :meth:`.from_file`.

        If ``construct`` is missing, parameters are forwarded to the regular
        :class:`.InstancedCanopyElement` constructor.

        Parameter ``d`` (dict):
            Dictionary containing parameters passed to the selected constructor.
            Unit fields are pre-processed with :func:`pinttr.interpret_units`.
        """
        # Interpret unit fields if any
        d_copy = pinttr.interpret_units(d, ureg=ureg)

        # Dispatch call based on 'construct' parameter
        construct = d_copy.pop("construct", None)

        if construct == "from_file":
            return cls.from_file(**d_copy)
        elif construct is None:
            return cls(**d_copy)
        else:
            raise ValueError(f"parameter 'construct': unsupported value '{construct}'")

    @staticmethod
    def convert(value):
        """
        Object converter method.

        If ``value`` is a dictionary, this method uses :meth:`from_dict` to
        create a :class:`.InstancedCanopyElement`.

        Otherwise, it returns ``value``.
        """
        if isinstance(value, dict):
            return InstancedCanopyElement.from_dict(value)

        return value

    # --------------------------------------------------------------------------
    #                        Kernel dictionary generation
    # --------------------------------------------------------------------------

    def bsdfs(self, ctx=None):
        """
        Return BSDF plugin specifications.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → dict:
            Return a dictionary suitable for merge with a :class:`.KernelDict`
            containing all the BSDFs attached to the shapes in the leaf cloud.
        """
        return self.canopy_element.bsdfs(ctx=ctx)

    def shapes(self, ctx=None):
        """
        Return shape plugin specifications.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → dict:
            A dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the shapes
            in the canopy.
        """
        return {
            self.canopy_element.id: {
                "type": "shapegroup",
                **self.canopy_element.shapes(ctx=ctx),
            }
        }

    def instances(self, ctx=None):
        """
        Return instance plugin specifications.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → dict:
            A dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing instances.
        """
        from mitsuba.core import ScalarTransform4f

        kernel_length = uck.get("length")

        return {
            f"{self.canopy_element.id}_instance_{i}": {
                "type": "instance",
                "group": {"type": "ref", "id": self.canopy_element.id},
                "to_world": ScalarTransform4f.translate(position.m_as(kernel_length)),
            }
            for i, position in enumerate(self.instance_positions)
        }

    def kernel_dict(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
        return {
            **self.bsdfs(ctx=ctx),
            **self.shapes(ctx=ctx),
            **self.instances(ctx=ctx),
        }
