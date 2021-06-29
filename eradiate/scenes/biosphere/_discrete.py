import itertools
import os
from copy import deepcopy
from typing import MutableMapping, Optional

import attr
import numpy as np
import pint
import pinttr

from ._core import BiosphereFactory, Canopy
from .canopy_element import CanopyElement, CanopyElementFactory
from ..core import SceneElement
from ..spectra import Spectrum, SpectrumFactory
from ... import validators
from ..._attrs import documented, get_doc, parse_docs
from ..._factory import BaseFactory
from ...contexts import KernelDictContext
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


@CanopyElementFactory.register("abstract_tree")
@parse_docs
@attr.s
class AbstractTree(CanopyElement):
    """
    A container class for abstract trees in discrete canopies.
    Holds a :class:`.LeafCloud` and the parameters characterizing a cylindrical
    trunk. The entire tree is described in local coordinates and can be placed
    in the scene using :class:`.InstancedCanopyElement`.

    The trunk starts at [0, 0, -0.1] and extends
    to [0, 0, trunk_height]. The trunk extends below ``z=0`` to avoid intersection
    issues at the intersection of the trunk and the ground the tree is usually placed on.

    The leaf cloud will by default be offset such that its local coordinate
    origin coincides with the upper end of the trunk. If this is not desired,
    e.g. the leaf cloud is centered around its coordinate origin and the trunk
    should not extend into it, the parameter ``leaf_cloud_extra_offset`` can be
    used to shift the leaf cloud **in addition** to the trunk's extent.

    The :meth:`.AbstractTree.from_dict` constructor will instantiate the trunk
    parameters based on dictionary specification and will forward the entry
    specifying the leaf cloud to :meth:`.LeafCloud.convert`.
    """

    leaf_cloud = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(LeafCloud.convert),
            validator=attr.validators.optional(attr.validators.instance_of(LeafCloud)),
        ),
        doc="Instanced leaf cloud. Can be specified as a dictionary, which will "
        "be interpreted by :meth:`.LeafCloud.from_dict`.",
        type=":class:`LeafCloud`",
        default="None",
    )

    trunk_height = documented(
        pinttr.ib(default=1.0 * ureg.m, units=ucc.deferred("length")),
        doc="Trunk height.\n\nUnit-enabled field (default: ucc[length]).",
        type="float",
        default="1.0 m",
    )

    trunk_radius = documented(
        pinttr.ib(default=0.1 * ureg.m, units=ucc.deferred("length")),
        doc="Trunk radius.\n\nUnit-enabled field (default: ucc[length]).",
        type="float",
        default="0.1 m",
    )

    trunk_reflectance = documented(
        attr.ib(
            default=0.5,
            converter=SpectrumFactory.converter("reflectance"),
            validator=[
                attr.validators.instance_of(Spectrum),
                validators.has_quantity("reflectance"),
            ],
        ),
        doc="Reflectance spectrum of the trunk. "
        "Must be a reflectance spectrum (dimensionless).",
        type=":class:`.Spectrum`",
        default="0.5",
    )

    leaf_cloud_extra_offset = documented(
        pinttr.ib(factory=lambda: [0, 0, 0], units=ucc.deferred("length")),
        doc="Additional offset for the leaf cloud. 3-vector.\n"
        "\n"
        "Unit-enabled field (default: ucc[length])",
        type="array-like",
        default="[0, 0, 0]",
    )

    def bsdfs(self, ctx=None):
        """
        Return BSDF plugin specifications.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → dict:
            Return a dictionary suitable for merge with a :class:`.KernelDict`
            containing all the BSDFs attached to the shapes
            in the abstract tree.
        """

        bsdfs_dict = self.leaf_cloud.bsdfs(ctx=ctx)

        bsdfs_dict[f"bsdf_{self.id}"] = {
            "type": "diffuse",
            "reflectance": self.trunk_reflectance.kernel_dict(ctx=ctx)["spectrum"],
        }

        return bsdfs_dict

    def shapes(self, ctx=None):
        """
        Return shape plugin specifications.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → dict:
            A dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the shapes
            in the abstract tree.
        """
        from mitsuba.core import ScalarTransform4f

        kernel_length = uck.get("length")

        kernel_height = self.trunk_height.m_as(kernel_length)
        kernel_radius = self.trunk_radius.m_as(kernel_length)

        leaf_cloud = self.leaf_cloud.translated(
            [0.0, 0.0, kernel_height] * kernel_length
            + self.leaf_cloud_extra_offset.to(kernel_length)
        )

        if ctx.ref:
            bsdf = {"type": "ref", "id": f"bsdf_{self.id}"}
        else:
            bsdf = self.bsdfs(ctx=ctx)[f"bsdf_{self.id}"]

        shapes_dict = leaf_cloud.shapes(ctx=ctx)

        shapes_dict[f"trunk_cyl_{self.id}"] = {
            "type": "cylinder",
            "bsdf": bsdf,
            "radius": kernel_radius,
            "p0": [0, 0, -0.1],
            "p1": [0, 0, kernel_height],
        }

        shapes_dict[f"trunk_cap_{self.id}"] = {
            "type": "disk",
            "bsdf": bsdf,
            "to_world": ScalarTransform4f.scale(kernel_radius)
            * ScalarTransform4f.translate(((0, 0, kernel_height))),
        }

        return shapes_dict

    @staticmethod
    def convert(value):
        """
        Object converter method.

        If ``value`` is a dictionary, this method uses :meth:`from_dict` to
        create an :class:`.AbstractTree`.

        Otherwise, it returns ``value``.
        """
        if isinstance(value, dict):
            return AbstractTree.from_dict(value)

        return value

    @classmethod
    def from_dict(cls, d):
        """
        Construct from a dictionary.


        Parameter ``d`` (dict):
            Dictionary containing parameters passed to the selected constructor.
            Unit fields are pre-processed with :func:`pinttr.interpret_units`.
        """

        # Interpret unit fields if any
        d_copy = pinttr.interpret_units(d, ureg=ureg)

        # pop the leaf cloud specs to avoid name collision with the
        # AbstractTree constructor
        leaf_cloud_dict = d_copy.pop("leaf_cloud")
        leaf_cloud = LeafCloud.convert(leaf_cloud_dict)

        return cls(leaf_cloud=leaf_cloud, **d_copy)


@parse_docs
@attr.s
class InstancedCanopyElement(SceneElement):
    """
    Specification a canopy element, alongside the locations of instances (*i.e.*
    clones) of it.

    .. admonition:: Class method constructors

       .. autosummary::

          from_file
          from_dict
    """

    canopy_element = documented(
        attr.ib(
            default=None,
            validator=attr.validators.optional(
                attr.validators.instance_of(CanopyElement)
            ),
            converter=CanopyElementFactory.convert,
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

        Parameter ``filename`` (str or PathLike):
            Path to the text file specifying the leaves in the canopy.
            Can be absolute or relative. Required (setting to ``None`` will
            raise an exception).

        Parameter ``canopy_element`` (:class:`.CanopyElement` or dict):
            :class:`.CanopyElement` to be instanced. If a dictionary is passed,
            if is interpreted by :meth:`.CanopyElement.from_dict`. If set to
            ``None``, an empty leaf cloud will be created.

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
            canopy_element = LeafCloud()
        else:
            canopy_element = CanopyElementFactory.convert(canopy_element)

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


@BiosphereFactory.register("discrete_canopy")
@parse_docs
@attr.s
class DiscreteCanopy(Canopy):
    """
    An abstract discrete canopy, consisting of one or several clouds of
    disk-shaped leaves. Each leaf cloud can be instanced arbitrarily. The
    produced canopy can be padded with more clones of itself using the
    :meth:`~.DiscreteCanopy.padded` method.

    The discrete canopy holds an :class:`InstancedCanopyElement` object, which
    in turn holds any class derived from :class:`CanopyElement`.

    .. admonition:: Tutorials

       * Practical usage ⇒ :ref:`sphx_glr_examples_generated_tutorials_biosphere_01_discrete_canopy.py`

    .. admonition:: Class method constructors

       .. autosummary::

          leaf_cloud_from_files
          homogeneous
    """

    instanced_canopy_elements = documented(
        attr.ib(
            factory=list,
            converter=lambda value: [
                InstancedCanopyElement.convert(x)
                for x in pinttr.util.always_iterable(value)
            ]
            if not isinstance(value, dict)
            else [InstancedCanopyElement.convert(value)],
        ),
        doc="List of :class:`.CanopyElement` defining the canopy. Can be "
        "initialised with a :class:`.InstancedCanopyElement`, which will be "
        "automatically wrapped into a list. Dictionary-based specifications are "
        "allowed as well.",
        type="list[:class:`.InstancedCanopyElement`]",
        default="[]",
    )

    def bsdfs(self, ctx=None):
        """
        Return BSDF plugin specifications.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → dict:
            Return a dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the BSDFs
            attached to the shapes in the canopy.
        """
        result = {}
        for instanced_canopy_element in self.instanced_canopy_elements:
            result = {**result, **instanced_canopy_element.bsdfs(ctx=ctx)}
        return result

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
        result = {}
        for instanced_canopy_element in self.instanced_canopy_elements:
            result = {**result, **instanced_canopy_element.shapes(ctx=ctx)}
        return result

    def instances(self):
        """
        Return instance plugin specifications.

        Returns → dict:
            A dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing instances.
        """
        result = {}
        for instanced_canopy_element in self.instanced_canopy_elements:
            result = {**result, **instanced_canopy_element.instances()}
        return result

    def kernel_dict(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
        if not ctx.ref:
            raise ValueError("'ctx.ref' must be set to True")

        result = {}
        for instanced_canopy_element in self.instanced_canopy_elements:
            result = {
                **result,
                **instanced_canopy_element.bsdfs(ctx=ctx),
                **instanced_canopy_element.shapes(ctx=ctx),
                **instanced_canopy_element.instances(ctx=ctx),
            }

        return result

    def padded(self, padding):
        """
        Return a copy of the current canopy padded with additional copies.

        Parameter ``padding`` (int):
            Amount of padding around the canopy. Must be positive or zero.
            The resulting padded canopy is a grid of
            :math:`2 \\times \\mathit{padding} + 1` copies.

        Returns → :class:`.DiscreteCanopy`:
            Padded copy.
        """
        if padding < 0:
            raise ValueError("padding must be >= 0")

        if padding == 0:
            return self

        # We'll return a copy
        result = deepcopy(self)

        # Convenience aliases
        config_length = ucc.get("length")
        x_size, y_size = result.size.m_as(config_length)[:2]
        padding_factors = np.array(list(range(-padding, padding + 1)))

        for instanced_canopy_element in result.instanced_canopy_elements:
            # More convenience aliases
            old_instance_positions = instanced_canopy_element.instance_positions.m_as(
                config_length
            )
            n_instances_per_cell = old_instance_positions.shape[0]
            # Allocate array for results
            new_instance_positions = np.empty(
                (len(padding_factors) ** 2 * n_instances_per_cell, 3)
            )

            for k, (x_offset_factor, y_offset_factor) in enumerate(
                itertools.product(padding_factors, padding_factors)
            ):
                # Set vector by which we will translate instance positions
                offset = np.array(
                    [x_size * x_offset_factor, y_size * y_offset_factor, 0.0]
                )

                # Compute new instances
                start_idx = k * n_instances_per_cell
                stop_idx = (k + 1) * n_instances_per_cell
                new_instance_positions[start_idx:stop_idx, :] = (
                    old_instance_positions[:, :] + offset
                )

            instanced_canopy_element.instance_positions = (
                new_instance_positions * config_length
            )

        # Update size
        result.size[:2] *= len(padding_factors)

        return result

    @classmethod
    def homogeneous(cls, id="homogeneous_discrete_canopy", **leaf_cloud_kwargs):
        """
        Generate a homogeneous discrete canopy.

        Parameter ``id`` (str):
            Canopy object ID.

        Parameter ``leaf_cloud_kwargs``:
            Keyword arguments forwarded to :meth:`.LeafCloud.cuboid`.

            .. note:: The leaf cloud's ID will be set to ``f"{id}_leaf_cloud"``.

        Returns → :class:`.DiscreteCanopy`:
            Created canopy object.
        """
        # Check parameters
        leaf_cloud_params = CuboidLeafCloudParams(**leaf_cloud_kwargs)
        config_length = ucc.get("length")
        size = [
            leaf_cloud_params.l_horizontal.m_as(config_length),
            leaf_cloud_params.l_horizontal.m_as(config_length),
            leaf_cloud_params.l_vertical.m_as(config_length),
        ] * config_length
        leaf_cloud_id = f"{id}_leaf_cloud"

        # Construct canopy
        return cls(
            id=id,
            size=size,
            instanced_canopy_elements=[
                InstancedCanopyElement(
                    instance_positions=[[0, 0, 0]],
                    canopy_element=LeafCloud.cuboid(
                        **leaf_cloud_kwargs, id=leaf_cloud_id
                    ),
                )
            ],
        )

    @classmethod
    def leaf_cloud_from_files(
        cls, id="discrete_canopy", size=None, leaf_cloud_dicts=None
    ):
        """
        Directly create a leaf cloud canopy from text file specifications.

         .. admonition:: CanopyElement dictionary format

           Each item of the ``leaf_cloud_dicts`` list shall have the following
           structure:

           .. code:: python

              {
                  "sub_id": "some_value",  # leaf cloud ID string part, optional if leaf_cloud_dicts has only 1 entry
                  "instance_filename": "some_path",  # path to instance specification file
                  "leaf_cloud_filename": "some_other_path", # path to leaf cloud specification file
                  "leaf_reflectance": 0.5,  # optional, leaf reflectance (default: 0.5)
                  "leaf_transmittance": 0.5,  # optional, leaf transmittance (default: 0.5)
              }


        Parameter ``id`` (str):
            Canopy ID.

        Parameter ``size`` (array-like):
            Canopy size as a 3-vector (in metres).

        Parameter ``leaf_cloud_dicts`` (list[dict]):
            List of dictionary specifying canopy elements and instances (see format
            above).

        Returns → :class:`.DiscreteCanopy`:
            Created canopy object.
        """
        # Check if required kwargs are provided (all args must be kwargs if we
        # want to use this constructor through from_dict())
        if size is None:
            raise ValueError(f"parameter 'size' is required")
        if leaf_cloud_dicts is None:
            raise ValueError(f"parameter 'leaf_cloud_dicts' is required")

        for param in [size, leaf_cloud_dicts]:
            if param is None:
                raise ValueError(f"parameter '{param}' is required")

        instanced_canopy_elements = []

        for leaf_cloud_dict in leaf_cloud_dicts:
            instance_filename = leaf_cloud_dict.get("instance_filename", None)

            leaf_cloud_params = {
                "filename": leaf_cloud_dict.get("leaf_cloud_filename", None),
                "leaf_reflectance": leaf_cloud_dict.get("leaf_reflectance", 0.5),
                "leaf_transmittance": leaf_cloud_dict.get("leaf_transmittance", 0.5),
            }

            sub_id = leaf_cloud_dict.get("sub_id", None)

            if sub_id is None:
                if len(leaf_cloud_dicts) > 1:
                    raise ValueError("parameter 'sub_id' must be set")
                leaf_cloud_params["id"] = f"{id}_leaf_cloud"
            else:
                leaf_cloud_params["id"] = f"{id}_{sub_id}_leaf_cloud"

            instanced_canopy_elements.append(
                InstancedCanopyElement.from_file(
                    filename=instance_filename,
                    canopy_element=LeafCloud.from_file(**leaf_cloud_params),
                )
            )

        return cls(
            id=id, size=size, instanced_canopy_elements=instanced_canopy_elements
        )

    @classmethod
    def from_dict(cls, d):
        """
        Construct from a dictionary. This function first queries for a
        ``construct`` parameter. If it is found, dictionary parameters are used
        to call another class method constructor:

        * ``homogeneous``: :meth:`.DiscreteCanopy.homogeneous`;
        * ``from_files``: :meth:`.DiscreteCanopy.from_files`.

        If ``construct`` is missing, parameters are forwarded to the regular
        :class:`.InstancedLeafCloud` constructor.

        Parameter ``d`` (dict):
            Dictionary containing parameters passed to the selected constructor.
            Unit fields are pre-processed with :func:`pinttr.interpret_units`.
        """
        # Interpret unit fields if any
        d_copy = pinttr.interpret_units(d, ureg=ureg)

        # Store padding value
        padding = d_copy.pop("padding", 0)

        # Dispatch call based on 'construct' parameter
        construct = d_copy.pop("construct", None)

        if construct == "homogeneous":
            result = cls.homogeneous(**d_copy)
        elif construct == "leaf_cloud_from_files":
            result = cls.leaf_cloud_from_files(**d_copy)
        elif construct is None:
            result = cls(**d_copy)
        else:
            raise ValueError(f"parameter 'construct': unsupported value '{construct}'")

        return result.padded(padding)
