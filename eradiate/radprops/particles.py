"""
Particles layers
"""
from abc import ABC, abstractmethod

import attr
import numpy as np
import pinttr
import xarray as xr
from pinttr.util import units_compatible
from scipy.stats import expon, norm

from .. import mode
from .._attrs import documented, parse_docs
from .._factory import BaseFactory
from .._presolver import PathResolver
from ..units import to_quantity
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg
from ..validators import is_positive

_presolver = PathResolver()


@attr.s
class LegendreExpansion:
    r"""Representation of a phase function as an expansion of Legendre
    polynomials, :math:`P_n (\cos \theta)`, where :math:`\theta` is the
    scattering angle.

    The first Legendre polynomials are given by:

    .. math::
        P_0 (x) = 1

    .. math::
        P_1 (x) = \frac{1}{2} (3x^2 - 1)

    .. math::
        P_2 (x) = \frac{1}{2} (5x^3 - 3x)


    .. rubric:: Constructor arguments / instance attributes

    ``coefficients`` (array):
       Coefficients of the Legendre polynomial expansion.
    """
    coefficients = attr.ib(converter=np.array)

    @classmethod
    def from_values(cls, values):
        """Expand phase function values into Legendre polynomials"""
        pass

    @classmethod
    def from_fourrier_series(cls, coefficients):
        """Create a Legendre polynomials expansion from a Fourrier series
        expansion."""
        pass

    @classmethod
    def from_spherical_harmonics(cls, coefficients):
        """Create a Legendre polynomials expansion from a spherical harmonics
        expansion.
        """
        pass


@parse_docs
@attr.s
class VerticalDistribution(ABC):
    r"""An abstract base class for particles vertical distributions.

    Vertical distributions help define particles layers.

    The particles layer is split into a number of divisions (sub-layers),
    wherein the particles fraction is evaluated.

    The vertical distribution is normalised so that:

    .. math::
        \sum_i f_i = 1

    where :math:`f_i` is the particles fraction in the layer division :math:`i`.
    """
    bottom = documented(
        pinttr.ib(
            units=ucc.deferred("length"),
            converter=pinttr.converters.to_units(ucc.deferred("length")),
            validator=pinttr.validators.has_compatible_units,
        ),
        doc="Layer bottom altitude.\nUnit-enabled field (default: ucc[length]).",
        type="float",
    )
    top = documented(
        pinttr.ib(
            units=ucc.deferred("length"),
            converter=pinttr.converters.to_units(ucc.deferred("length")),
            validator=pinttr.validators.has_compatible_units,
        ),
        doc="Layer top altitude.\nUnit-enabled field (default: ucc[length]).",
        type="float",
    )

    def __attrs_post_init__(self):
        if self.bottom >= self.top:
            raise ValueError("bottom altitude must be lower than top altitude")

    @classmethod
    def from_dict(cls, d):
        """Initialise a :class:`VerticalDistribution` from a dictionary."""
        return cls(**d)

    @property
    @abstractmethod
    def fractions(self):
        """Returns a callable that evaluates the particles fractions in the
        layer, given an array of altitude values."""
        pass

    @staticmethod
    def _normalise(x):
        """Scale the values so that their sum is 1.

        Parameter ``x`` (array):
            Values to normalise.

        Returns → array:
            Normalised array.
        """
        _norm = np.sum(x)
        if _norm > 0.0:
            return x / _norm
        else:
            raise ValueError(f"Cannot normalise fractions because the norm is " f"0.")


class VerticalDistributionFactory(BaseFactory):
    """This factory constructs objects whose classes are derived from
    :class:`VerticalDistribution`

    .. admonition:: Registered factory members
       :class: hint

       .. factorytable::
          :factory: VerticalDistributionFactory
    """

    _constructed_type = VerticalDistribution
    registry = {}


@VerticalDistributionFactory.register("uniform")
@parse_docs
@attr.s
class Uniform(VerticalDistribution):
    r"""Uniform vertical distribution

    The uniform probability distribution function is:

    .. math::
        f(z) = \frac{1}{z_{\rm top} - z_{\rm bottom}}, \quad
        z \in [z_{\rm top}, z_{\rm bottom}]

    where :math:`z_{\rm top}` and :math:`z_{\rm bottom}` are the layer top and bottom
    altitudes, respectively.
    """

    @property
    def fractions(self):
        def eval(z):
            """
            Parameter ``z`` (:class:`pint.Quantity`):
                Altitude values.

            Return → array:
                Particles fractions.
            """
            if (self.bottom <= z).all() and (z <= self.top).all():
                x = z.magnitude
                return self._normalise(np.ones(len(x)))
            else:
                raise ValueError(
                    f"Altitude values do not lie between layer "
                    f"bottom ({self.bottom}) and top ({self.top}) "
                    f"altitudes. Got {z}."
                )

        return eval


@VerticalDistributionFactory.register("exponential")
@parse_docs
@attr.s
class Exponential(VerticalDistribution):
    r"""Exponential vertical distribution.

    The exponential probability distribution function is:

    .. math::
        f(z) = \lambda  \exp \left( -\lambda z \right)

    where :math:`\lambda` is the rate parameter and :math:`z` is the altitude.
    """
    rate = documented(
        pinttr.ib(
            units=ucc.deferred("collision_coefficient"),
            default=None,
            converter=attr.converters.optional(
                pinttr.converters.to_units(ucc.deferred("collision_coefficient"))
            ),
            validator=attr.validators.optional(pinttr.validators.has_compatible_units),
        ),
        doc="Rate parameter of the exponential distribution. If ``None``, "
        "set to the inverse of the layer thickness."
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="float",
        default="``None``",
    )

    def __attrs_post_init__(self):
        if self.rate is None:
            self.rate = 1.0 / (self.top - self.bottom)

    @property
    def fractions(self):
        def eval(z):
            if (self.bottom <= z).all() and (z <= self.top).all():
                x = z.magnitude
                loc = self.bottom.to(z.units).magnitude
                scale = (1.0 / self.rate).to(z.units).magnitude
                f = expon.pdf(x=x, loc=loc, scale=scale)
                return self._normalise(f)
            else:
                raise ValueError(
                    f"Altitude values do not lie between layer "
                    f"bottom ({self.bottom}) and top ({self.top}) "
                    f"altitudes. Got {z}."
                )

        return eval


@VerticalDistributionFactory.register("gaussian")
@parse_docs
@attr.s
class Gaussian(VerticalDistribution):
    r"""Gaussian vertical distribution.

    The Gaussian probability distribution function is:

    .. math::
        f(z) = \frac{1}{2 \pi \sigma}
        \exp{\left[
            -\frac{1}{2}
            \left( \frac{z - \mu}{\sigma} \right)^2
        \right]}

    where :math:`\mu` is the mean of the distribution and :math:`\sigma` is
    the standard deviation of the distribution.
    """
    mean = documented(
        pinttr.ib(
            units=ucc.deferred("length"),
            default=None,
            converter=attr.converters.optional(
                pinttr.converters.to_units(ucc.deferred("length"))
            ),
            validator=attr.validators.optional(pinttr.validators.has_compatible_units),
        ),
        doc="Mean (expectation) of the distribution. "
        "If ``None``, set to the middle of the layer."
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="float",
        default="``None``",
    )
    std = documented(
        pinttr.ib(
            units=ucc.deferred("length"),
            default=None,
            converter=attr.converters.optional(
                pinttr.converters.to_units(ucc.deferred("length"))
            ),
            validator=attr.validators.optional(pinttr.validators.has_compatible_units),
        ),
        doc="Standard deviation of the distribution. If ``None``, set to one "
        "sixth of the layer thickness so that half the layer thickness "
        "equals three standard deviations."
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="float",
        default="``None``",
    )

    def __attrs_post_init__(self):
        if self.mean is None:
            self.mean = (self.bottom + self.top) / 2.0
        if self.std is None:
            self.std = (self.top - self.bottom) / 6.0

    @property
    def fractions(self):
        def eval(z):
            if (self.bottom <= z).all() and (z <= self.top).all():
                x = z.magnitude
                loc = self.mean.to(z.units).magnitude
                scale = self.std.to(z.units).magnitude
                f = norm.pdf(x=x, loc=loc, scale=scale)
                return self._normalise(f)
            else:
                raise ValueError(
                    f"Altitude values do not lie between layer "
                    f"bottom ({self.bottom}) and top ({self.top}) "
                    f"altitudes. Got {z}."
                )

        return eval


@VerticalDistributionFactory.register("array")
@parse_docs
@attr.s
class Array(VerticalDistribution):
    """Flexible vertical distribution specified either by an array of values,
    or by a :class:`DataArray` object.
    """

    values = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(np.array),
            validator=attr.validators.optional(attr.validators.instance_of(np.ndarray)),
        ),
        doc="Particles fractions values on a regular altitude mesh starting "
        "from the layer bottom and stopping at the layer top altitude.",
        type="array",
        default="``None``",
    )
    data_array = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(xr.DataArray),
            validator=attr.validators.optional(
                attr.validators.instance_of(xr.DataArray)
            ),
        ),
        doc="Particles vertical distribution data array. Fraction as a function"
        " of altitude (``z``).",
        type=":class:`xarray.DataArray`",
        default="``None``",
    )

    @data_array.validator
    def check(self, attribute, value):
        if value is not None:
            if not "z" in value.coords:
                raise ValueError("Attribute 'data_array' must have a 'z' " "coordinate")
            else:
                try:
                    units = ureg.Unit(value.z.units)
                    if not units_compatible(units, ureg.Unit("m")):
                        raise ValueError(
                            f"Coordinate 'z' of attribute "
                            f"'data_array' must have units"
                            f"compatible with m^-1 (got {units})."
                        )
                except AttributeError:
                    raise ValueError(
                        "Coordinate 'z' of attribute 'data_array' " "must have units."
                    )

    method = documented(
        attr.ib(
            default="linear", converter=str, validator=attr.validators.instance_of(str)
        ),
        doc="Method to interpolate the data. This parameter is passed to "
        ":meth:`xarray.DataArray.interp`.",
        type="str",
        default='``"linear"``',
    )

    def __attrs_post_init__(self):
        if self.values is None and self.data_array is None:
            raise ValueError("You must specify 'values' or 'data_array'.")
        elif self.values is not None and self.data_array is not None:
            raise ValueError(
                "You cannot specify both 'values' and " "'data_array' simultaneously."
            )
        elif self.data_array is None:
            self.data_array = xr.DataArray(
                data=self.values,
                coords={
                    "z": (
                        "z",
                        np.linspace(
                            start=self.bottom.to("m").magnitude,
                            stop=self.top.to("m").magnitude,
                            num=len(self.values),
                        ),
                        {"units": "m"},
                    )
                },
                dims=["z"],
            )
        elif self.data_array is not None:
            min_z = to_quantity(self.data_array.z.min(keep_attrs=True))
            if min_z < self.bottom:
                raise ValueError(
                    f"Minimum altitude value in data_array "
                    f"({min_z}) is smaller than bottom altitude "
                    f"({self.bottom})."
                )

            max_z = to_quantity(self.data_array.z.max(keep_attrs=True))
            if max_z > self.top:
                raise ValueError(
                    f"Minimum altitude value in data_array "
                    f"({max_z}) is smaller than top altitude"
                    f"({self.top})."
                )

    @property
    def fractions(self):
        def eval(z):
            x = z.to(self.data_array.z.units).magnitude
            f = self.data_array.interp(
                coords={"z": x}, method=self.method, kwargs=dict(fill_value=0.0)
            )
            return self._normalise(f.values)

        return eval


@parse_docs
@attr.s
class ParticlesLayer:
    """1D particles layer."""

    dataset = documented(
        attr.ib(
            default="aeronet_desert",
            validator=attr.validators.instance_of(str),
        ),
        doc="Particles radiative properties dataset.",
        type="str",
        default='``"aeronet_desert"``',
    )

    bottom = documented(
        pinttr.ib(
            default=ureg.Quantity(0.0, ureg.km),
            converter=pinttr.converters.to_units(ucc.deferred("length")),
            validator=[
                is_positive,
                pinttr.validators.has_compatible_units,
            ],
            units=ucc.deferred("length"),
        ),
        doc="Bottom altitude of the particles layer."
        "\n"
        "Unit-enabled field (default: ucc[length])",
        type="float",
        default="0 km",
    )

    top = documented(
        pinttr.ib(
            units=ucc.deferred("length"),
            default=ureg.Quantity(1.0, ureg.km),
            converter=pinttr.converters.to_units(ucc.deferred("length")),
            validator=[
                is_positive,
                pinttr.validators.has_compatible_units,
            ],
        ),
        doc="Top altitude of the particles layer."
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="float",
        default="1 km.",
    )

    tau_550 = documented(
        pinttr.ib(
            units=ucc.deferred("dimensionless"),
            default=ureg.Quantity(0.2, ureg.dimensionless),
            validator=[
                is_positive,
                pinttr.validators.has_compatible_units,
            ],
        ),
        doc="Extinction optical thickness at the wavelength of 550 nm."
        "\n"
        "Unit-enabled field (default: ucc[dimensionless]).",
        type="float",
        default="0.2",
    )

    vert_dist = documented(
        attr.ib(
            default={"type": "uniform"},
            validator=attr.validators.instance_of((dict, VerticalDistribution)),
        ),
        doc="Particles vertical distribution.",
        type="dict or :class:`VerticalDistribution`",
        default=":class:`Uniform`",
    )

    n_layers = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(int),
            validator=attr.validators.optional(attr.validators.instance_of(int)),
        ),
        doc="Number of layers inside the particles layer."
        "If ``None``, set to a different value based on the vertical "
        "distribution type (see table below).\n"
        "\n"
        ".. list-table::\n"
        "   :widths: 1 1\n"
        "   :header-rows: 1\n"
        "\n"
        "   * - Vertical distribution type\n"
        "     - Number of layers\n"
        "   * - :class:`Uniform`\n"
        "     - 1\n"
        "   * - :class:`Exponential`\n"
        "     - 8\n"
        "   * - :class:`Gaussian`\n"
        "     - 16\n"
        "   * - :class:`Array`\n"
        "     - 32\n"
        "\n",
        type="int",
        default="``None``",
    )

    def __attrs_post_init__(self):
        # update the keys 'bottom' and 'top' in vertical distribution config
        if isinstance(self.vert_dist, dict):
            d = self.vert_dist
            d.update({"bottom": self.bottom, "top": self.top})
            self.vert_dist = VerticalDistributionFactory.convert(d)

        # determine layers number based on vertical distribution type
        if self.n_layers is None:
            if isinstance(self.vert_dist, Uniform):
                self.n_layers = 1
            elif isinstance(self.vert_dist, Exponential):
                self.n_layers = 8
            elif isinstance(self.vert_dist, Gaussian):
                self.n_layers = 16
            elif isinstance(self.vert_dist, Array):
                self.n_layers = 32

    @property
    def z_layer(self):
        """Returns the layer altitudes."""
        bottom = self.bottom.to("km").magnitude
        top = self.top.to("km").magnitude
        z_level = np.linspace(start=bottom, stop=top, num=self.n_layers + 1)
        z_layer = (z_level[:-1] + z_level[1:]) / 2.0
        return ureg.Quantity(z_layer, "km")

    @property
    def fractions(self):
        """Returns the particles fractions in the layer."""
        return self.vert_dist.fractions(self.z_layer)

    @property
    def phase(self):
        """Return phase function Legendre polynomials expansion

        Returns → :class:`.Legendre`:
            Phase functions Legendre polynomials expansion coefficients
        """
        return LegendreExpansion([1.0, 0.0])

    def eval_albedo(self, spectral_ctx):
        """
        Evaluate albedo given a spectral context.

        Returns → :class:`pint.Quantity`:
            Particles layer albedo.
        """
        wavelength = spectral_ctx.wavelength
        ds = xr.open_dataset(_presolver.resolve(path=self.dataset + ".nc"))
        interpolated_albedo = ds.albedo.interp(w=wavelength)
        albedo = to_quantity(interpolated_albedo)
        albedo_array = albedo * np.ones(self.n_layers)
        return albedo_array

    def eval_sigma_t(self, spectral_ctx):
        """
        Evaluate extinction coefficient given a spectral context.

        Returns → :class:`pint.Quantity`:
            Particles layer extinction coefficient.
        """
        wavelength = spectral_ctx.wavelength
        ds = xr.open_dataset(_presolver.resolve(path=self.dataset + ".nc"))
        interpolated_sigma_t = ds.sigma_t.interp(w=wavelength)
        sigma_t = to_quantity(interpolated_sigma_t)
        sigma_t_array = sigma_t * self.fractions
        normalised_sigma_t_array = self._normalise_to_tau(
            ki=sigma_t_array,
            dz=(self.top - self.bottom) / self.n_layers,
            tau=self.tau_550,
        )
        return normalised_sigma_t_array

    def eval_sigma_a(self, spectral_ctx):
        """
        Evaluate absorption coefficient given a spectral context.

        Returns → :class:`pint.Quantity`:
            Particles layer absorption coefficient.
        """
        return self.eval_sigma_t(spectral_ctx) - self.eval_sigma_a(spectral_ctx)

    def eval_sigma_s(self, spectral_ctx):
        """
        Evaluate scattering coefficient given a spectral context.

        Returns → :class:`pint.Quantity`:
            Particles layer scattering coefficient.
        """
        return self.eval_sigma_t(spectral_ctx) * self.eval_albedo(spectral_ctx)

    @classmethod
    def from_dict(cls, d):
        """Initialise a :class:`ParticlesLayer` from a dictionary."""
        return cls(**d)

    @classmethod
    def convert(cls, value):
        """Object converter method.

        If ``value`` is a dictionary, this method forwards it to
        :meth:`from_dict`. Otherwise, it returns ``value``.
        """
        if isinstance(value, dict):
            return cls.from_dict(value)

        return value

    def to_dataset(self, spectral_ctx):
        """Return a dataset that holds the radiative properties of the
        particles layer.

        Returns → :class:`xarray.Dataset`:
            Particles layer radiative properties dataset.
        """
        sigma_t = self.eval_sigma_t(spectral_ctx)
        albedo = self.eval_albedo(spectral_ctx)
        z_layer = self.z_layer
        wavelength = spectral_ctx.wavelength
        return xr.Dataset(
            data_vars={
                "sigma_t": (
                    ("w", "z_layer"),
                    np.atleast_2d(sigma_t.magnitude),
                    {
                        "standard_name": "extinction_coefficient",
                        "long_name": "extinction coefficient",
                        "units": sigma_t.units,
                    },
                ),
                "albedo": (
                    ("w", "z_layer"),
                    np.atleast_2d(albedo.magnitude),
                    {
                        "standard_name": "albedo",
                        "long_name": "albedo",
                        "units": albedo.units,
                    },
                ),
            },
            coords={
                "z_layer": (
                    "z_layer",
                    z_layer.magnitude,
                    {
                        "standard_name": "layer_altitude",
                        "long_name": "layer altitude",
                        "units": z_layer.units,
                    },
                ),
                "w": (
                    "w",
                    [wavelength.magnitude],
                    {
                        "standard_name": "wavelength",
                        "long_name": "wavelength",
                        "units": wavelength.units,
                    },
                ),
            },
        )

    @staticmethod
    @ureg.wraps(ret="km^-1", args=("km^-1", "km", ""), strict=False)
    def _normalise_to_tau(ki, dz, tau):
        r"""Normalise extinction coefficient values :math:`k_i` so that:

        .. math::

            \sum_i k_i \Delta z = \tau_{550}

        where :math:`tau` is the particles layer optical thickness.

        Parameter ``ki`` (array):
            Extinction coefficients values [km^-1].

        Parameter ``dz`` (array):
            Layer divisions thickness [km].

        Parameter ``tau`` (float):
            Layer optical thickness (dimensionless).

        Returns → array:
            Normalised extinction coefficients.
        """
        return ki * tau / (np.sum(ki) * dz)
