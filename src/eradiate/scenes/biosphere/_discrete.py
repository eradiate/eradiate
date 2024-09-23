from __future__ import annotations

import itertools
import typing as t
from collections.abc import MutableMapping
from copy import deepcopy

import attrs
import numpy as np
import pint
import pinttr

from ._core import Canopy, InstancedCanopyElement, biosphere_factory
from ._leaf_cloud import CuboidLeafCloudParams, LeafCloud
from ...attrs import define, documented
from ...units import unit_context_config as ucc


def _instanced_canopy_elements_converter(value):
    """
    Special converter for the DiscreteCanopy.instanced_canopy_elements field.
    """
    if isinstance(value, MutableMapping):
        value["type"] = "instanced"
    return biosphere_factory.convert(value)


@define(eq=False, slots=False)
class DiscreteCanopy(Canopy):
    """
    Discrete canopy scene element [``discrete_canopy``].

    A discrete canopy composed of one or several :class:`.CanopyElement`
    instances. The underlying implementation holds a list of
    :class:`.InstancedCanopyElement`, allowing for arbitrary cloning of each
    :class:`.CanopyElement`. The produced canopy can be padded with clones of
    itself using the :meth:`~.DiscreteCanopy.padded_copy` method.

    .. admonition:: Class method constructors

       .. autosummary::

          homogeneous
          leaf_cloud_from_files
          padded
    """

    # --------------------------------------------------------------------------
    #                                   Fields
    # --------------------------------------------------------------------------

    instanced_canopy_elements: list[InstancedCanopyElement] = documented(
        attrs.field(
            factory=list,
            converter=lambda value: [
                _instanced_canopy_elements_converter(x)
                for x in pinttr.util.always_iterable(value)
            ]
            if not isinstance(value, MutableMapping)
            else [_instanced_canopy_elements_converter(value)],
            validator=attrs.validators.deep_iterable(
                member_validator=attrs.validators.instance_of(InstancedCanopyElement)
            ),
        ),
        doc="List of :class:`.CanopyElement` defining the canopy. Can be "
        "initialized with a :class:`.InstancedCanopyElement`, which will be "
        "automatically wrapped into a list. Dictionary-based specifications are "
        "allowed as well.",
        type="list of :class:`.InstancedCanopyElement`",
        default="[]",
    )

    # --------------------------------------------------------------------------
    #                          Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def _template_bsdfs(self) -> dict:
        result = {}
        for element in self.instanced_canopy_elements:
            result.update(element._template_bsdfs)
        return result

    @property
    def _template_shapes(self) -> dict:
        result = {}
        for element in self.instanced_canopy_elements:
            result.update(element._template_shapes)
        return result

    @property
    def _template_instances(self) -> dict:
        result = {}
        for element in self.instanced_canopy_elements:
            result.update(element._template_instances)
        return result

    @property
    def template(self) -> dict:
        result = {}

        for element in self.instanced_canopy_elements:
            result.update(
                {
                    **element._template_bsdfs,
                    **element._template_shapes,
                    **element._template_instances,
                }
            )

        return result

    @property
    def _params_instances(self) -> dict:
        result = {}
        for element in self.instanced_canopy_elements:
            result.update(element._params_instances)
        return result

    @property
    def params(self) -> dict:
        result = {}

        for element in self.instanced_canopy_elements:
            result.update(
                {
                    **element._params_bsdfs,
                    **element._params_shapes,
                    **element._params_instances,
                }
            )

        return result

    # --------------------------------------------------------------------------
    #                                  Padding
    # --------------------------------------------------------------------------

    def padded_copy(self, padding: int) -> DiscreteCanopy:
        """
        Return a copy of the current canopy padded with additional copies.

        Parameters
        ----------
        padding : int
            Amount of padding around the canopy. Must be positive or zero.
            The resulting padded canopy is a grid of
            :math:`2 \\times \\mathit{padding} + 1` copies.

        Returns
        -------
        .DiscreteCanopy
            Padded copy of self.
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

    # --------------------------------------------------------------------------
    #                               Constructors
    # --------------------------------------------------------------------------

    @classmethod
    def padded(cls, padding: int = 0, **kwargs) -> DiscreteCanopy:
        """
        Create a discrete canopy and pad it with copies of itself. Keyword
        arguments are forwarded to the default constructor.

        Parameters
        ----------
        padding : int
            Amount of padding around the canopy. Must be positive or zero.
            The resulting padded canopy is a grid of
            :math:`2 \\times \\mathit{padding} + 1` copies.

        **kwargs``
            Keyword arguments forwarded to the :class:`.DiscreteCanopy`
            constructor.

        Returns
        -------
        :class:`.DiscreteCanopy`
            Padded discrete canopy.
        """
        return cls(**kwargs).padded_copy(padding)

    @classmethod
    def homogeneous(
        cls,
        padding: int = 0,
        id: str = "homogeneous_discrete_canopy",
        **leaf_cloud_kwargs,
    ) -> DiscreteCanopy:
        """
        Generate a homogeneous discrete canopy, possibly padded with copies of
        itself.

        Parameters
        ----------
        padding : int
            Amount of padding around the canopy. Must be positive or zero.
            The resulting padded canopy is a grid of
            :math:`2 \\times \\mathit{padding} + 1` copies.

        id : str, default: "homogeneous_discrete_canopy"
            Canopy object ID.

        **leaf_cloud_kwargs
            Keyword arguments forwarded to :meth:`.LeafCloud.cuboid`.

        Returns
        -------
        .DiscreteCanopy
            Created canopy object.

        Notes
        -----
        The leaf cloud's ID will be set to ``f"{id}_leaf_cloud"``.
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
        ).padded_copy(padding)

    @classmethod
    def leaf_cloud_from_files(
        cls,
        size: pint.Quantity,
        leaf_cloud_dicts: list[t.MutableMapping],
        padding: int = 0,
        id: str = "discrete_canopy",
    ):
        """
        Directly create a leaf cloud canopy from text file specifications,
        possibly padded with copies of itself.

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

        Parameters
        ----------
        size : quantity
            Canopy size as a 3-vector (in metres).

        leaf_cloud_dicts : list of dict
            List of dictionary specifying canopy elements and instances (see
            format above).

        padding : int, default: 0
            Amount of padding around the canopy. Must be positive or zero.
            The resulting padded canopy is a grid of
            :math:`2 \\times \\mathit{padding} + 1` copies.

        id : str, default: "discrete_canopy"
            Canopy ID.

        Returns
        -------
        .DiscreteCanopy
            Created canopy object.
        """
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
            id=id,
            size=size,
            instanced_canopy_elements=instanced_canopy_elements,
        ).padded_copy(padding)
