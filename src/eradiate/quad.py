"""Utility components for quadrature rules."""

from __future__ import annotations

from enum import Enum

import attrs
import numpy as np

from .attrs import define, documented
from .util.misc import str_summary_numpy


class QuadType(Enum):
    """Quadrature rule type flags."""

    GAUSS_LEGENDRE = "gauss_legendre"
    GAUSS_LOBATTO = "gauss_lobatto"


@define(eq=False, frozen=True)
class Quad:
    """
    A data class storing information about a quadrature rule. Nodes and weights
    are defined in the [-1, 1] interval. The reference interval can be changed
    using the ``interval`` argument of the :meth:`.eval_nodes` and
    :meth:`.integrate` functions.
    """

    type: QuadType = documented(
        attrs.field(converter=QuadType, repr=lambda x: str(x)),
        doc="Quadrature type. If a string is passed, it is converted to a "
        ":class:`.QuadType`.",
        type=":class:`.QuadType`",
    )

    nodes: np.ndarray = documented(
        attrs.field(converter=np.array, repr=str_summary_numpy),
        doc="Quadrature rule nodes.",
        type="ndarray",
    )

    weights: np.ndarray = documented(
        attrs.field(converter=np.array, repr=str_summary_numpy),
        doc="Quadrature rule weights.",
        type="ndarray",
    )

    @nodes.validator
    @weights.validator
    def _nodes_weights_validator(self, attribute, value):
        if self.nodes.shape != self.weights.shape:
            raise ValueError(
                f"while validating {attribute.name}: nodes and weights arrays "
                f"must have the same shape, got nodes.shape = {self.nodes.shape} "
                f"and weights.shape = {self.weights.shape}"
            )

    def pretty_repr(self) -> str:
        return f"{self.type.value}, {self.nodes.size} points"

    @classmethod
    def gauss_legendre(cls, n: int) -> Quad:
        """
        Initialize a :class:`.Quad` instance with Gauss-Legendre nodes and
        weights.

        Parameters
        ----------
        n : int
            Number of quadrature points.

        Returns
        -------
        :class:`.Quad`
            Gauss-Legendre quadrature definition.
        """
        from mitsuba.scalar_rgb.quad import gauss_legendre

        nodes, weights = gauss_legendre(n)
        return cls(
            type=QuadType.GAUSS_LEGENDRE,
            nodes=np.array(nodes, dtype=float),
            weights=np.array(weights, dtype=float),
        )

    @classmethod
    def gauss_lobatto(cls, n: int) -> Quad:
        """
        Initialize a :class:`.Quad` instance with Gauss-Lobatto nodes and
        weights.

        Parameters
        ----------
        n : int
            Number of quadrature points.

        Returns
        -------
        :class:`.Quad`
            Gauss-Lobatto quadrature definition.
        """
        from mitsuba.scalar_rgb.quad import gauss_lobatto

        nodes, weights = gauss_lobatto(n)
        return cls(
            type=QuadType.GAUSS_LOBATTO,
            nodes=np.array(nodes, dtype=float),
            weights=np.array(weights, dtype=float),
        )

    @classmethod
    def new(cls, type: str | QuadType, n: int) -> Quad:
        """
        Initialize a :class:`.Quad` instance of the specified type.

        Parameters
        ----------
        type : str or .QuadType
            Quadrature rule type. If a string is passed, it is converted to
            a :class:`.QuadType`.

        n : int
            Number of quadrature points.

        Returns
        -------
        :class:`.Quad`
            Quadrature definition.
        """
        type = QuadType(type)

        if type is QuadType.GAUSS_LEGENDRE:
            return cls.gauss_legendre(n)

        elif type is QuadType.GAUSS_LOBATTO:
            return cls.gauss_lobatto(n)

        else:
            raise ValueError(f"unknown quadrature type '{type}'")

    def eval_nodes(
        self, interval: tuple[float, float] | None = None
    ) -> np.typing.ArrayLike:
        """
        Compute nodes scaled to a specific interval.

        Parameters
        ----------
        interval :  tuple of float, optional
            Interval for which nodes are to be scaled as a 2-tuple. If ``None``,
            the default [-1, 1] is used.

        Returns
        -------
        ndarray
            Scaled node values.
        """
        if interval is None:
            return self.nodes
        a, b = interval
        return 0.5 * (a + b + (b - a) * self.nodes)

    def integrate(
        self, values: np.typing.ArrayLike, interval: tuple[float, float] | None
    ) -> float:
        """
        Evaluate quadrature rule, accounting for interval scaling.

        Parameters
        ----------
        values : ndarray
            Function values at quadrature nodes.

        interval : tuple of float, optional
            Interval on which the integral is being computed as a 2-tuple.
            If ``None``, the default [-1, 1] is used.

        Returns
        -------
        float
            Quadrature evaluation for the specified interval.
        """

        weighted_sum = float(np.dot(self.weights, values))

        if interval is None:
            return weighted_sum
        else:
            return 0.5 * (interval[1] - interval[0]) * weighted_sum

    @property
    def str_summary(self) -> str:
        """
        Return a summarized representation of the current instance.

        Returns
        -------
        str
            Instance summary.
        """
        return f"Quad(type={QuadType(self.type)}, n={len(self.nodes)})"
