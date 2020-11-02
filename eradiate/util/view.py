"""A tool for handling and plotting subsets of data."""

import warnings
from abc import ABC, abstractmethod

import attr
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import eradiate.kernel

from ..util import frame
from .xarray import eo_dataarray


@attr.s
class SampledAdapter(ABC):
    """Adapter for BRDF backends requiring sampling for their evaluation."""

    @abstractmethod
    def evaluate(self, wo, wi, wavelength):
        r"""Retrieve the value of the BSDF for a given set of incoming and
        outgoing directions and wavelength.

        Parameter ``wo`` (array)
            Direction of outgoing radiation, given as a 2-vector of zenith
            and azimuth values in degrees.

        Parameter ``wi`` (array)
            Direction of outgoing radiation, given as a 2-vector of zenith
            and azimuth values in degrees.

        Parameter ``wavelength`` (float)
            Wavelength to query the BSDF plugin at.

        Returns → float
            Evaluated BRDF value.
        """
        pass


@attr.s
class MitsubaBSDFPluginAdapter(SampledAdapter):
    r"""Wrapper class around :class:`~mitsuba.render.BSDF` plugins.
    Holds a BSDF plugin and exposes an evaluate method, taking care of internal
    necessities for evaluating the plugin.

    Instance attributes:
        ``bsdf`` (:class:`~mitsuba.render.BSDF`): Mitsuba BSDF plugin
    """
    bsdf = attr.ib()

    @bsdf.validator
    def _check_bsdf(self, attribute, value):
        if not isinstance(value, eradiate.kernel.render.BSDF):
            raise TypeError(
                f"This class must be instantiated with a Mitsuba"
                f"BSDF plugin. Found: {type(value)}"
            )

    def evaluate(self, wo, wi, wavelength):
        from eradiate.kernel.render import SurfaceInteraction3f, BSDFContext
        ctx = BSDFContext()
        si = SurfaceInteraction3f()
        wi = np.deg2rad(wi)
        si.wi = frame.angles_to_direction(wi[0], wi[1])
        si.wavelengths = [wavelength]
        if len(wo) == 2:
            wo = np.deg2rad(wo)
            wo = frame.angles_to_direction(wo[0], wo[1])

        return self.bsdf.eval(ctx, si, wo)[0] / wo[2]


def plane(hdata, phi=0):
    """Extract a plane data set from a hemispherical data set.
    This method will select data on a plane oriented along the azimuth direction
    ``phi`` and its complementary ``phi`` + 180°, and stitch the two subsets
    together.

    Data at azimuth angle ``phi`` will be mapped to positive zenith values, while
    data at ``phi`` + 180° will be mapped to negative zenith values.

    .. note::
        If ``hdata`` contains other non-angular dimensions (*e.g.* wavelength),
        they will persist in the returned array.

    Parameter ``hdata`` (:class:`~xarray.DataArray`)
        Data set from which to the create the plane data set.

    Parameter ``phi`` (float)
        Viewing azimuth angle to orient the plane view. If set to None,
        phi will be set to be equal to ``phi_i``, providing the principal plane.

    Returns → :class:`~xarray.DataArray`
        Extracted plane data set for the requested azimuth angle.
    """

    # Check if source is a hemispherical data set (2 angular dimensions)
    if not hdata.ert.is_hemispherical(exclude_scalar_dims=True):
        raise ValueError("hdata must be a hemispherical data set")

    # Support for angular dimension naming conventions
    theta_dim = hdata.ert.get_angular_dim("theta_o")

    # Retrieve values for positive half-plane
    theta_pos = hdata.coords[theta_dim]
    values_pos = hdata.ert.sel(
        phi_o=phi,
        theta_o=theta_pos,
        method="nearest"
    )

    # Retrieve values for negative half-plane   
    theta_neg = hdata.coords[theta_dim][1:]
    values_neg = hdata.ert.sel(
        phi_o=(phi + 180.) % 360.,
        theta_o=theta_neg,
        method="nearest"
    )
    values_neg = values_neg.assign_coords({theta_dim: -theta_neg})  # Transform zeniths to negative values
    values_neg = values_neg.loc[{theta_dim: sorted(values_neg.coords[theta_dim].values)}]  # Reorder data

    # Combine negative and positive half-planes
    arr = xr.concat((values_neg, values_pos), dim=theta_dim)
    arr.coords[theta_dim].attrs = hdata.coords[theta_dim].attrs  # We don't forget to copy metadata

    return arr

def pplane(bhdata, theta_i=None, phi_i=None):
    """Extract a principal plane view from a bi-hemispherical data set. This
    operation, in practice, consists in extracting a hemispherical view based on
    the incoming direction angles ``theta_i`` and ``phi_i``, then applying the
    plane view extraction function :func:`plane` with ``phi = phi_i``.

    Parameter ``bhdata`` (:class:`~xarray.DataArray`):
        Bi-hemispherical data set (with four angular directions).

    Parameter ``theta_i`` (float or None):
        Incoming zenith angle. If `None`, select the first available value.

    Parameter ``phi_i`` (float or None):
        Incoming azimuth angle. If `None`, select the first available value.

    Returns → :class:`~xarray.DataArray`
        Extracted principal plane data set for the requested incoming angular
        configuration.
    """
    if not bhdata.ert.is_bihemispherical():
        raise ValueError("bhdata must be a bi-hemispherical data set")

    if theta_i is None:
        theta_i_dim = bhdata.ert.get_angular_dim("theta_i")

        try:  # if theta_i_dim is an array
            theta_i = float(bhdata.coords[theta_i_dim][0])
        except TypeError:  # if theta_i_dim is scalar
            theta_i = float(bhdata.coords[theta_i_dim])

    if phi_i is None:
        phi_i_dim = bhdata.ert.get_angular_dim("phi_i")

        try:  # if phi_i_dim is an array
            phi_i = float(bhdata.coords[phi_i_dim][0])
        except TypeError:  # if phi_i_dim is scalar
            phi_i = float(bhdata.coords[phi_i_dim])

    return plane(bhdata.ert.sel(theta_i=theta_i, phi_i=phi_i), phi=phi_i)


def bihemispherical_data_from_plugin(source, sza, saa, vza_res, vaa_res, wavelength):
    """Sample a BSDF plugin to create a data array containing scattering information for a given incoming radiation direction
    and a given resolution for the outgoing radiation.
    
    Parameter ``source`` (:class:`~mitsuba.render.BSDF`):
        Eradiate kernel bsdf plugin.
        
    Parameter ``sza`` (float):
        Illumination zenith angle.
    
    Parameter ``saa`` (float):
        Illumination azimuth angle.

    Parameter ``vza_res`` (float):
        Viewing zenith angle resolution

    Parameter ``vaa_res`` (float):
        Viewing azimuth angle resolution.

    Parameter ``wavelength`` (float):
        Wavelength to evaluate the bsdf plugin at.

    Returns → :class:`~xarray.DataArray`
        Data array holding scattering values into the observed hemisphere for a given incoming radiation
        configuration and wavelength.
    """
    bsdf = MitsubaBSDFPluginAdapter(source)
    wi = (sza, saa)
    vza = np.linspace(0, 90, int(90 / vza_res))
    vaa = np.linspace(0, 360, int(360 / vaa_res))

    data = np.zeros((len(vaa), len(vza)))
    for i, theta in enumerate(vza):
        for j, phi in enumerate(vaa):
            data[j, i] = bsdf.evaluate((theta, phi), wi, wavelength)

    data = np.expand_dims(data, [0, 1, 4])
    arr = eo_dataarray(data, [sza], [saa], vza, vaa, wavelength, "hemispherical")

    return arr


@xr.register_dataarray_accessor("ert")
class EradiateAccessor:
    """Custom :class:`xarray.DataArray` accessor to process Eradiate results.
    """

    _ANGLE_CONVENTIONS = {
        "local": {
            "theta_i": "theta_i",
            "phi_i": "phi_i",
            "theta_o": "theta_o",
            "phi_o": "phi_o"
        },
        "eo_scene": {
            "theta_i": "sza",
            "phi_i": "saa",
            "theta_o": "vza",
            "phi_o": "vaa"
        }
    }

    def get_angular_dim(self, dim):
        """Return the angle dimension corresponding to the angular naming convention in the data."""
        if dim not in ("theta_i", "phi_i", "theta_o", "phi_o"):
            raise ValueError("dim must be in ('theta_i', 'phi_i', 'theta_o', 'phi_o')")

        try:
            convention = self._obj.attrs["angle_convention"]
        except KeyError:
            raise KeyError("No angle naming convention is set. Cannot convert dimension names.")

        return self._ANGLE_CONVENTIONS[convention][dim]

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def _num_angular_dimensions(self, exclude_scalar_dims=False):
        """Determine the number of angular dimensions by their unit.
        Recognized angular dimensions are given in one of the following units:

        'radians', 'rad', 'degrees', 'deg'
        """
        num_angdims = 0
        for dim in self._obj.dims:
            if self._obj.coords[dim].attrs["unit"] in ("radians", "rad", "degrees", "deg"):
                if exclude_scalar_dims:
                    if len(self._obj.coords[dim]) > 1:
                        num_angdims += 1
                else:
                    num_angdims += 1
        return num_angdims

    def is_hemispherical(self, exclude_scalar_dims=False):
        return self._num_angular_dimensions(exclude_scalar_dims) == 2

    def is_bihemispherical(self, exclude_scalar_dims=False):
        return self._num_angular_dimensions(exclude_scalar_dims) == 4

    def _adjust_kwargs_for_convention(self, **kwargs):
        """Translates the given keyword arguments to the respective naming convention in the
        DataArray. If all requested dimensions are present in the array, no change is performed.
        Otherwise the angle convention attribute of the array is looked up and the
        requested dimensions are translated accordingly.

        Returns → dict
            Keyword arguments adjusted for the angular naming convention in the DataArray.
        """
        for arg in kwargs:
            if arg not in self._obj.dims:
                break
        else:
            return kwargs

        try:
            convention = self._obj.attrs["angle_convention"]
        except KeyError:
            raise KeyError("No angle convention was set. Cannot identify data nomenclature.")
        try:
            convention_map = self._ANGLE_CONVENTIONS[convention]
        except KeyError:
            raise KeyError(f"Unknown angle naming convention: {convention}")

        newargs = dict()
        for oldkey in kwargs:
            if oldkey in convention_map:
                newargs[convention_map[oldkey]] = kwargs[oldkey]
            else:
                newargs[oldkey] = kwargs[oldkey]

        return newargs

    def sel(self, **kwargs):
        """Wraps the :meth:`xarray.DataArray.sel` method to account for
        different naming conventions for angle dimensions.
        If the presented args are present in the :class:`~xarray.DataArray`,
        they are passed on unchanged.
        Otherwise a naming convention lookup is performed and the corresponding
        dimension names are used.

        Returns → :class:`xarray.DataArray`
            View to the original xarray with data selected as given by the
            arguments.
        """
        new_kwargs = self._adjust_kwargs_for_convention(**kwargs)
        return self._obj.sel(**new_kwargs)

    def drop_sel(self, **kwargs):
        """Wraps the :meth:`xarray.DataArray.drop_sel` method to account for
        the angular dimension naming conventions.
        """
        new_kwargs = self._adjust_kwargs_for_convention(**kwargs)
        return self._obj.drop_sel(**new_kwargs)

    def plot(self, kind=None, ax=None, title="", cmap="BuPu_r", cbar_kwargs={}, **kwargs):
        """Create a plot suitable for Eradiate result data.

        .. note::

            Dimensions of illumination angles and wavelength must be collapsed by selecting a
            value from them, if they contain more than one value.

        Parameter ``kind`` (str):
            Kind of the plot to be generated.
            If `None`, redirect call to `xarray.DataArray.plot`.

            For hemispherical datasets (*i.e.* with 2 angular dimensions), polar
            plots can be obtained using the following values:

            - ``polar_pcolormesh``: use :func:`matplotlib.pyplot.pcolormesh`;
            - ``polar_contourf``: use :func:`matplotlib.pyplot.contourf`.

        Parameter ``ax`` (:class:`~matplotlib.axes`):
            Optional Axes object to integrate the view into a custom plotting script.

        Parameter ``title`` (str):
            Optional title for the generated plot.

        Parameter ``cmap`` (str or matplotlib colormap):
            Optional colormap to style the plot.

        Parameter ``cbar_kwargs`` (dict):
            Optional parameters to tweak the color bar.

        Parameter ``kwargs``:
            Other keyword arguments passed to the underlying plotting routine.
        """

        with xr.set_options(cmap_sequential=cmap):
            if kind not in {"polar_pcolormesh", "polar_contourf"}:
                return self._obj.plot(ax=ax, **kwargs)

            if self._num_angular_dimensions(exclude_scalar_dims=True) != 2:
                warnings.warn(f"Dimensions {self._obj.dims} unsuitable for plotting, "
                              f"redirecting to xarray's plotting function")
                return self._obj.plot(ax=ax, **kwargs)

        theta_i_dim = self.get_angular_dim("theta_i")
        phi_i_dim = self.get_angular_dim("phi_i")
        try:
            len_theta = len(self._obj.coords[theta_i_dim])
        except TypeError:
            len_theta = 0

        try:
            len_phi = len(self._obj.coords[phi_i_dim])
        except TypeError:
            len_phi=0

        if len_theta > 1 or len_phi > 1:
            raise IndexError("Angular configuration for incoming light is ambiguous."
                             "Please select an angular configuration before plotting.")

        try:
            len_wavelength = len(self._obj.coords["wavelength"])
        except TypeError:
            len_wavelength = 0
        if len_wavelength > 1:
            raise IndexError("Multiple wavelengths are present in the dataset. Please select one wavelength before plotting.")

        data = np.squeeze(self._obj.values)

        if ax is None:
            ax = plt.gca(projection="polar")

        theta_o_dim = self.get_angular_dim("theta_o")
        theta_o_angles = self._obj[theta_o_dim].values
        phi_o_dim = self.get_angular_dim("phi_o")
        phi_o_angles = self._obj[phi_o_dim].values

        r, th = np.meshgrid(
            np.deg2rad(theta_o_angles),
            np.deg2rad(phi_o_angles)
        )

        if kind == "polar_pcolormesh":
            cmap_data = ax.pcolormesh(
                th, r, np.transpose(data),
                cmap=cmap, shading="nearest", **kwargs
            )

        elif kind == "polar_contourf":
            cmap_data = ax.contourf(
                th, r, np.transpose(data),
                cmap=cmap, **kwargs
            )

        else:
            raise ValueError(f"unsupported plot kind {kind}")

        ticks, labels = generate_ticks(5, (0, np.pi / 2.))
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.grid(True)
        plt.colorbar(cmap_data, **cbar_kwargs)
        plt.title("\n".join([title, self._obj._title_for_slice()]))

        return ax


def generate_ticks(num_ticks, limits):
    """Generates ticks and their respective tickmarks.

    Parameter ``num_ticks`` (int):
        Number of ticks to generate, including the limits
        of the given range

    Parameter ``limits`` (list[float]):
        List of two values, limiting the ticks inclusive

    Returns → list, list:
        - Values for the ticks
        - Tick values converted to degrees as string tickmarks
    """

    step_width = float(limits[1] - limits[0]) / (num_ticks - 1)

    steps = [limits[0] + step_width * i for i in range(num_ticks)]
    marks = [f"{i / np.pi * 180}°" for i in steps]

    return steps, marks
