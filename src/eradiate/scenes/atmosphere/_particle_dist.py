"""
Particle distributions.

Particle distributions define how the particle number fraction varies with
altitude. The particle layer is split into a number of divisions
(sub-layers) wherein the particle number fraction is evaluated.

Notes
-----
Particle distributions are not normalized. The parent caller is responsible
for normalizing returned values.
"""

import typing as t
from abc import ABC, abstractmethod

import attrs
import numpy as np
import scipy.interpolate

from ..._factory import Factory
from ...attrs import define, documented

particle_distribution_factory = Factory()
particle_distribution_factory.register_lazy_batch(
    [
        ("UniformParticleDistribution", "uniform", {}),
        ("ExponentialParticleDistribution", "exponential", {}),
        ("GaussianParticleDistribution", "gaussian", {}),
        ("ArrayParticleDistribution", "array", {}),
        ("InterpolatorParticleDistribution", "interpolator", {}),
    ],
    cls_prefix="eradiate.scenes.atmosphere._particle_dist",
)


@define
class ParticleDistribution(ABC):
    """
    Abstract base class for particle distributions used to define particle
    layers.

    In practice, particle distributions are callables with the signature
    ``f(x: np.typing.ArrayLike) -> np.ndarray`` and are evaluated over the
    interval [0, 1].
    """

    @abstractmethod
    def __call__(self, x: np.typing.ArrayLike) -> np.ndarray:
        pass


@define
class UniformParticleDistribution(ParticleDistribution):
    r"""
    Uniform particle distribution [``uniform``].

    Returns values given by the uniform PDF

    .. math::
       f : x \mapsto \left\{
           \begin{array}{ll}
               \frac{1}{b - a} & \mathrm{if} \ x \in [a, b] \\
               0 & \mathrm{otherwise}
           \end{array}
       \right.

    where :math:`a = \mathtt{bounds[0]}` and :math:`b = \mathtt{bounds[1]}`.
    """

    bounds: np.ndarray = documented(
        attrs.field(
            default=[0.0, 1.0], converter=lambda x: np.atleast_1d(np.squeeze(x))
        ),
        type="array",
        init_type="array-like, optional",
        default="[0, 1]",
        doc="Bounds of the distribution's interval.",
    )

    @bounds.validator
    def _bounds_validator(self, attribute, value):
        if len(value) != 2:
            raise ValueError(
                f"while validating '{attribute.name}': passed array must have "
                "exactly 2 elements"
            )

        if value[1] <= value[0]:
            raise ValueError(
                f"while validating '{attribute.name}': bounds must be sorted in "
                "ascending order "
            )

    def __call__(self, x: np.typing.ArrayLike) -> np.ndarray:
        return np.where(
            np.logical_or(x < self.bounds[0], x > self.bounds[1]),
            np.zeros_like(x),
            np.full_like(x, 1.0 / (self.bounds[1] - self.bounds[0])),
        )


@define(init=False)
class ExponentialParticleDistribution(ParticleDistribution):
    r"""
    Exponential particle distribution [``exponential``].

    Returns values given by the normalized exponential PDF

    .. math::
       f : x \mapsto \dfrac{\lambda e^{-\lambda x}}{1 - e^{-\lambda}}

    where :math:`\lambda = \mathtt{rate}`.

    Parameters
    ----------
    rate : float, optional, default: 5.0
        Decay rate :math:`\lambda`. The default value ensures a 99.3% decay on
        the [0, 1] interval. Mutually exclusive with `scale`.

    scale : float, optional
        Scale :math:`\beta = 1 / \lambda`. Mutually exclusive with `rate`.

    Fields
    ------
    rate : float
        Decay rate :math:`\lambda`.
    """

    rate: float = attrs.field(default=5.0, converter=float)

    @rate.validator
    def _rate_validator(self, attribute, value):
        if value <= 0:
            raise ValueError(
                f"while validating {attribute.name}: rate must be strictly positive"
            )

    def __init__(self, rate: float = None, scale: float = None):
        if rate is None and scale is None:
            self.__attrs_init__()
        elif scale is None and rate is not None:
            self.__attrs_init__(rate)
        elif scale is not None and rate is None:
            self.__attrs_init__(1.0 / scale)
        else:
            raise ValueError(
                "while initializing ExponentialParticleDistribution: "
                "setting both rate and scale is not allowed"
            )

    def __call__(self, x: np.typing.ArrayLike) -> np.ndarray:
        return np.where(
            np.logical_or(x < 0.0, x > 1.0),
            np.zeros_like(x),
            np.exp(-x * self.rate) * self.rate / (1.0 - np.exp(-self.rate)),
        )


@define
class GaussianParticleDistribution(ParticleDistribution):
    r"""
    Gaussian particle distribution [``gaussian``].

    Returns values given by the Gaussian PDF

    .. math::
       f : x \mapsto \frac{1}{2 \pi \cdot \sigma}
           \exp \left[
             -\frac{1}{2}
             \left( \frac{x - \mu}{\sigma} \right)^2
           \right]

    where :math:`\mu = \mathtt{mean}` and :math:`\sigma = \mathtt{std}`.
    """

    mean: float = documented(
        attrs.field(default=0.5, converter=float),
        type="float",
        init_type="float, optional",
        default="0.5",
        doc="Mean of the Gaussian PDF. The default value places the mean in "
        "the middle of the particle layer (at :math:`x = 0.5`).",
    )

    std: float = documented(
        attrs.field(default=0.5 / 3, converter=float),
        type="float",
        init_type="float, optional",
        default="1/6",
        doc="Standard deviation of the Gaussian PDF. The default value is "
        "such that the integral of the Gaussian PDF over the "
        r":math:`[\mu - 0.5, \mu + 0.5]` interval is about 99.7% (3σ).",
    )

    def __call__(self, x: np.typing.ArrayLike) -> np.ndarray:
        return np.where(
            np.logical_or(x < 0.0, x > 1.0),
            np.zeros_like(x),
            np.exp(-0.5 * np.square((x - self.mean) / self.std))
            / (self.std * np.sqrt(2.0 * np.pi)),
        )


@define
class ArrayParticleDistribution(ParticleDistribution):
    """
    Particle distribution specified by an array of values [``array``].
    """

    values: np.typing.ArrayLike = documented(
        attrs.field(converter=np.array, kw_only=True),
        type="ndarray",
        init_type="array-like",
        doc="An array of particle fraction values.",
    )

    @values.validator
    def _values_validator(self, attribute, value):
        if value.ndim != 1:
            raise ValueError(
                f"while validating {attribute.name}: only 1D arrays are allowed"
            )

        if len(value) < 2:
            raise ValueError(
                f"while validating {attribute.name}: array must have at least 2 "
                "elements"
            )

    coords: np.ndarray = documented(
        attrs.field(
            default=attrs.Factory(
                lambda x: np.arange(
                    0.5 / len(x.values),
                    1,
                    1 / len(x.values),
                ),
                takes_self=True,
            ),
            converter=np.array,
        ),
        type="ndarray",
        init_type="array-like, optional",
        doc="Coordinates to which passed values are mapped. This array must "
        "have the same shape as ``values``. The default value positions values "
        "at the centers of a regular grid with nodes defined "
        "by :code:`np.linspace(0, 1, len(values))`.",
    )

    @coords.validator
    def _coords_validator(self, attribute, value):
        if value.shape != self.values.shape:
            raise ValueError(
                f"while validating '{attribute.name}': coordinate and value "
                "array shapes must be the same"
            )

    method: str = documented(
        attrs.field(
            default="linear",
            converter=str,
            validator=attrs.validators.in_(
                {
                    "linear",
                    "nearest",
                    "nearest-up",
                    "zero",
                    "slinear",
                    "quadratic",
                    "cubic",
                    "previous",
                    "next",
                }
            ),
        ),
        type="str",
        init_type='{ "linear", "nearest", "nearest-up", "zero", "slinear", '
        '"quadratic", "cubic", "previous", "next" }',
        default='"linear"',
        doc="Interpolation method. See :class:`scipy.interpolate.interp1d` "
        "(*kind*) for more information.",
    )

    extrapolate: str = documented(
        attrs.field(
            default="zero",
            converter=str,
            validator=attrs.validators.in_({"zero", "nearest", "method", "nan"}),
        ),
        type="str",
        init_type='{ "zero", "nearest", "method", "nan" }',
        default='"zero"',
        doc="Extrapolation method used when evaluation is requested outside of "
        r":math:`[\mathtt{coords[0]}, \mathtt{coords[-1]}]`. "
        "See :class:`scipy.interpolate.interp1d` (*fill_value*) for more "
        "information. Settings map as follows:\n"
        "\n"
        ".. list-table::\n\n"
        "   * - ``ArrayParticleDistribution`` (`extrapolate`)\n"
        "     - ``interp1d`` (`fill_value`)\n"
        '   * - ``"zero"``\n'
        "     - ``0.0``\n"
        '   * - ``"nearest"``\n'
        "     - ``(values[0], values[-1])``\n"
        '   * - ``"method"``\n'
        "     - ``extrapolate``\n"
        '   * - ``"nan"``\n'
        "     - ``np.nan``\n",
    )

    def __call__(self, x: np.typing.ArrayLike) -> np.ndarray:
        if self.extrapolate == "zero":
            fill_value = 0.0
        elif self.extrapolate == "nearest":
            fill_value = (self.values[0], self.values[-1])
        elif self.extrapolate == "method":
            fill_value = "extrapolate"
        else:
            fill_value = np.nan

        # TODO: This is unsafe, values of x outside of [0, 1] should return 0
        f = scipy.interpolate.interp1d(
            self.coords,
            self.values,
            kind=self.method,
            bounds_error=False,
            fill_value=fill_value,
        )
        return f(x)


@define
class InterpolatorParticleDistribution(ParticleDistribution):
    """
    A flexible particle distribution which redirects its calls to an
    encapsulated callable [``interpolator``].
    """

    interpolator: t.Callable[[np.typing.ArrayLike], np.ndarray] = documented(
        attrs.field(validator=attrs.validators.is_callable, kw_only=True),
        type="callable",
        doc="A callable with signature "
        ":code:`f(x: np.typing.ArrayLike) -> np.ndarray`. Typically, "
        "a :class:`scipy.interpolate.interp1d`.",
    )

    def __call__(self, x: np.typing.ArrayLike) -> np.ndarray:
        # TODO: This is unsafe, values of x outside of [0, 1] should return 0
        return self.interpolator(x)
