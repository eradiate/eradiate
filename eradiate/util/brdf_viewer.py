"""Helper classes to visualise BRDF data."""

from abc import ABC, abstractmethod

import attr
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import eradiate
import eradiate.kernel
from ..util import frame


@attr.s
class BRDFAdapter(ABC):
    """Abstract class for different B[RS]DF backends."""
    pass


class SampledAdapter(BRDFAdapter):
    """Adapter for BRDF backends requiring sampling for their evaluation."""

    @abstractmethod
    def evaluate(self, wo, wi, wavelength):
        r"""Retrieve the value of the BSDF for a given set of incoming and
        outgoing directions and wavelength.

        Parameter ``wo`` (array)
            Direction of outgoing radiation, given as a 2-vector of :math:`\theta`
            and :math:`\phi` values in degrees.

        Parameter ``wi`` (array)
            Direction of outgoing radiation, given as a 2-vector of :math:`\theta`
            and :math:`\phi` values in degrees.

        Parameter ``wavelength`` (float)
            Wavelength to query the BSDF plugin at.

        Returns → float
            Evaluated BRDF value.
        """
        pass


class GriddedAdapter(BRDFAdapter):
    """Adapter for BRDF backends consisting of gridded data."""
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
            wo = frame.angles_to_direction(wo[0], wo[1])

        return self.bsdf.eval(ctx, si, wo)[0] / wo[2]

    def sample(self):
        raise NotImplementedError("Coming soon.")


@attr.s
class DataArrayAdapter(GriddedAdapter):
    r"""Wrapper class around xarray formatted data from Eradiate solvers.
    Holds gridded data obtained through radiative transfer computation

    If data along the :code:`\phi_o` axis does not contain data for 0° and 360° one value
    is copied to the other position to complete the data set for interpolation.

    This class expects the xarray data to be formatted in the following way:

    - The array has 5 dimensions, named and ordered like this:
      :code:`['theta_i', 'phi_i', 'theta_o', 'phi_o', 'wavelength']`
    - The phi_o dimension must contain values for at least one of the two values
      0° and 360°. If only one is present, it is copied to the other value for better plotting

    """

    data = attr.ib()

    @data.validator
    def _check_data(self, attribute, value):
        if not isinstance(value, xr.DataArray):
            raise TypeError(
                f"This class must be instantiated with a xarray.DataArray"
                f"Found: {type(value)}"
            )
        required_dims = ["theta_i", "phi_i", "theta_o", "phi_o", "wavelength"]
        for dim in required_dims:
            if dim not in self.data.dims:
                raise ValueError(
                    f"Required data dimension {dim} not present in data!")

    def __attrs_post_init__(self):
        if 0. not in self.data.phi_o and 360. not in self.data.phi_o:
            raise ValueError("Data contains data for neither 0° nor 360°!")
        elif 0. not in self.data.phi_o:
            self.data = self._copy_phi_o_values(360., 0.)
        elif 360. not in self.data.phi_o:
            self.data = self._copy_phi_o_values(0., 360.)

    def _copy_phi_o_values(self, origin, target):
        r"""Constructs a new xarray, containing one extra value in the
        ":math:`\phi_o`" dimension.
        Data from :math:`\phi_o` == origin is copied into
        :math:`\phi_o` == target.
        """

        coords = {dim: self.data.coords[dim].values for dim in self.data.dims}
        coords["phi_o"] = np.append(coords["phi_o"], target)

        empty = np.zeros([len(coords[dim]) for dim in self.data.dims])
        data_new = xr.DataArray(
            empty,
            [coords[dim] for dim in self.data.dims],
            self.data.dims
        )

        theta_i_ind = \
            xr.DataArray(np.arange(len(self.data.coords["theta_i"])),
                         dims=["theta_i"])
        phi_i_ind = \
            xr.DataArray(np.arange(len(self.data.coords["phi_i"])),
                         dims=["phi_i"])
        theta_o_ind = \
            xr.DataArray(np.arange(len(self.data.coords["theta_o"])),
                         dims=["theta_o"])
        phi_o_ind = \
            xr.DataArray(np.arange(len(self.data.coords["phi_o"])),
                         dims=["phi_o"])
        wavelength_ind = \
            xr.DataArray(np.arange(len(self.data.coords["wavelength"])),
                         dims=["wavelength"])

        data_new[
            theta_i_ind,
            phi_i_ind,
            theta_o_ind,
            phi_o_ind,
            wavelength_ind
        ] = self.data[
            theta_i_ind,
            phi_i_ind,
            theta_o_ind,
            phi_o_ind,
            wavelength_ind
        ]
        data_new = xr.where(
            (data_new.coords["phi_o"] == target),
            self.data.sel(phi_o=origin),
            data_new
        )

        return data_new

    def _plotting_data(self, wi, wavelength):
        r"""Return data required for plotting

        Parameter ``wi`` (numpy.ndarray)
            Pair of (theta,phi) values to select the proper values from the gridded data
            Must match exactly the data points in the DataArray

        Parameter ``wavelength`` (float)
            Wavelength to select the proper values from the gridded data
            Must match exactly the data points in the DataArray

        Returns → numpy.ndarray, numpy.ndarray, xarray.DataArray:
            - :math:`\theta_o` values
            - :math:`\phi_o` values
            - data used for plot generation
        """
        theta_i, phi_i = wi

        if theta_i not in self.data.coords['theta_i'] or \
                phi_i not in self.data.coords['phi_i'] or \
                wavelength not in self.data.coords['wavelength']:
            raise ValueError("Selected values don't align with data grid! "
                             "Wavelength and incoming direction cannot be "
                             "interpolated.")

        data = self.data.sel(theta_i=theta_i, phi_i=phi_i, wavelength=wavelength).data
        return self.data.coords['theta_o'], \
               self.data.coords['phi_o'], \
               data


@attr.s
class BRDFView(ABC):
    r"""The BRDFViewer class is a utility to visualize bidirectional reflection
    distribution functions. It can read data from Mitsuba plugins as well as
    BRDF files computed with Eradiate itself.

    Directions (wi, wo) are represented as 2-vectors of :math:`(\theta,\phi)`
    values in degrees.

    It can create polar plots as well as linear plots of the principal plane.

    Instance attributes:
        ``brdf`` (:class:`BRDF`):
            Source of scattering data to be evaluated and plotted
        ``wi`` (:class:`float`):
            Incoming direction of radiation in spherical (in degrees) coordinates
        ``azm`` (:class:`float`):
            Azimuth angle resolution, can be given as a number of
            steps or a resolution in degrees
        ``zen`` (:class:`float`):
            Zenith angle resolution, can be given as a number of
            steps or a resolution in degrees
        ``wavelength`` (:class:`float`):
            Wavelength in nanometers for evaluation
            can be omitted, if the brdf is specified with no spectral dependency
    """

    _brdf = attr.ib(init=False)
    _wi = attr.ib(default=[0, 0])
    _azm = attr.ib(default=np.linspace(0., np.pi * 2., 73))  # One point per 5°
    _zen = attr.ib(default=np.linspace(0., np.pi / 2., 19))  # One point per 5°
    _wavelength = attr.ib(default=550.)
    _z = attr.ib(init=False)

    @property
    def wavelength(self):
        """Wavelength in nanometers to evaluate the BSDF at.
        For BSDFs that are specified without spectral dependency, this value
        can be left at its default value of 550 nm.
        """
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wavelength):
        self._wavelength = wavelength

    @property
    def azm(self):
        """Azimuth resolution, can be set in steps, using the azm_steps setter
        or as a resolution using the azm_res setter
        """
        return self._azm

    @azm.setter
    def azm(self, angles):
        """Directly set the azimuth angles with an array of angles in radians."""
        self._azm = angles

    @property
    def azm_steps(self):
        """The azimuth resolution in steps.
        Note that the underlying data structure accounts for 0° as well as
        360° and take this into account when choosing the number of steps.
        """
        return len(self._azm)

    @azm_steps.setter
    def azm_steps(self, steps):
        self._azm = np.linspace(0, np.pi * 2, steps)

    @property
    def azm_res(self):
        """Azimuth resolution as angular resolution
        If the given value yields a non integer number of steps, the number of 
        steps is rounded up to the next integer.
        """
        return 360 / (len(self._azm) - 1)

    @azm_res.setter
    def azm_res(self, res):
        steps = int(np.ceil(360.0 / float(res)))
        self.azm_steps = steps + 1

    @property
    def zen(self):
        """Zenith resolution, can be set in steps, using the azm_steps setter
        or as a resolution using the azm_res setter
        """
        return self._zen

    @zen.setter
    def zen(self, angles):
        """Directly set the zenith angles with an array of angles in radians."""
        self._zen = angles

    @property
    def zen_steps(self):
        """The zenith resolution in steps.
        Note that the underlying data structure accounts for 0° as well as
        90° and take this into account when choosing the number of steps.
        """
        return len(self._zen)

    @zen_steps.setter
    def zen_steps(self, steps):
        self._zen = np.linspace(0, np.pi / 2.0, steps)

    @property
    def zen_res(self):
        """Zenith resolution as angular resolution
        If the given value yields a non integer number of steps, the number of 
        steps is rounded up to the next integer.
        """
        return 360 / (len(self._zen) - 1)

    @zen_res.setter
    def zen_res(self, res):
        steps = int(np.ceil(90.0 / float(res)))
        self.zen_steps = steps + 1

    @property
    def wi(self):
        """Direction of the incoming radiation.
        Can be specified in cartesian or spherical coordinates (in degrees).
        """
        return self._wi

    @wi.setter
    def wi(self, wi):
        if len(wi) == 2:
            self._wi = wi
        elif len(wi) == 3:
            norm = np.linalg.norm(wi)
            if norm == 0:
                raise ValueError("Incoming direction vector cannot have "
                                 "length 0!")
            else:
                self._wi = np.rad2deg(
                    frame.direction_to_angles(wi / np.linalg.norm(wi))
                )

    @property
    def brdf(self):
        return self._brdf

    @brdf.setter
    def brdf(self, source):
        """The brdf can be specified with an eradiate.kernel.render.BSDF object
        or with a string pointing to a data file holding scattering data
        retrieved by simulating radiative transfer with Eradiate

        Parameter ``source`` (object) or (str):
            Source to load the BRDF from.
            If a eradiate.kernel.render.BSDF object is given, the BSDPlugin wrapper
            will be instantiated.
        """

        if isinstance(source, eradiate.kernel.render.BSDF):
            self._brdf = MitsubaBSDFPluginAdapter(source)

        elif isinstance(source, xr.DataArray):
            self._brdf = DataArrayAdapter(source)

        elif isinstance(source, str):
            raise TypeError("Loading BRDF data files is not supported yet.")

        else:
            raise TypeError(f"Non supported BRDF source type: {type(source)}")

    @abstractmethod
    def evaluate(self):
        """Evaluate the BRDF object, filling the data structures necessary for
        the desired plot.

        The :class:`BRDFView` classes discriminate two types of
        :class:`BRDFAdapter`:

        - :class:`SampledAdapter` s expose a method
          :meth:`~SampledAdapter.evaluate()` where each data point in the plot
          is queried individually and the adapter handles the retrieval of
          values for arbitrary incoming and outgoing directions
        - :class:`GriddedAdapter` s expose a method
          :meth:`~GriddedAdapter._plotting_data` which overrides the viewer's
          settings for resolution, setting it to match exactly the number of
          data points in the gridded data
        """
        pass

    @abstractmethod
    def plot(self):
        """Set up the plot as specified by the explicit class, returning the
        Axes object for display
        """
        pass

    @abstractmethod
    def to_xarray(self):
        """Return the relevant data for plotting (theta, phi, z) into an xarray
        Theta and Phi values are stored in the coordinates of the z data.
        Output can, among other things, be accessed like this:

        data[theta=np.pi/2., phi=np.pi]

        Returns → xarray.DataArray:
            Data obtained from evaluate
        """
        pass


class HemisphericalView(BRDFView):
    """Creates a polar plot of scattering into the hemisphere which holds the incoming
    radiation and the surface normal.
    """

    def evaluate(self):

        if isinstance(self.brdf, GriddedAdapter):
            thetas, phis, data = self.brdf._plotting_data(
                self.wi, self.wavelength)
            self.zen = np.deg2rad(thetas)
            self.azm = np.deg2rad(phis)
            self._z = data
            self.r, self.th = np.meshgrid(self.zen, self.azm)

        elif isinstance(self.brdf, SampledAdapter):
            self.r, self.th = np.meshgrid(self.zen, self.azm)
            self._z = np.zeros(np.shape(self.r))

            for i in range(np.shape(self.r)[0]):
                for j in range(np.shape(self.r)[1]):
                    wo = [self.r[i][j], self.th[i][j]]
                    val = self.brdf.evaluate(wo, self.wi, self.wavelength)
                    self._z[i][j] = val

        else:
            raise TypeError(f"Unsupported adapter {type(self.brdf)}!")

    def plot(self, ax=None, mode="pcolormesh"):
        """Output the data to a polar plot.

        Parameter ``ax`` (:class:`~matplotlib.axes.Axes`):
            Axis object for attaching the plot
            If not set, the current axes object will be obtained from matplotlib

        Parameter ``mode`` (str):
            Plotting command used to create the plot.
            Accepted values are:

            - ``pcolormesh`` (uses :func:`matplotlib.pyplot.pcolormesh`)
            - ``contourf`` (uses :func:`matplotlib.pyplot.contourf`)

        Returns → :class:`~matplotlib.axes.Axes`:
            An Axes object for use in custom matplotlib setups
        """
        if ax is None:
            ax = plt.gca(projection="polar")

        if isinstance(self.brdf, GriddedAdapter):
            if mode == "pcolormesh":
                cmap_data = ax.pcolormesh(self.th, self.r, np.transpose(self._z), cmap="BuPu_r")
            elif mode == "contourf":
                cmap_data = ax.contourf(self.th, self.r, np.transpose(self._z), cmap="BuPu_r")
            else:
                raise ValueError(f"unsupported plot mode {mode}")
        else:
            if mode == "pcolormesh":
                cmap_data = ax.pcolormesh(self.th, self.r, self._z, cmap="BuPu_r")
            elif mode == "contourf":
                cmap_data = ax.contourf(self.th, self.r, self._z, cmap="BuPu_r")
            else:
                raise ValueError(f"unsupported plot mode {mode}")

        ticks, labels = generate_ticks(5, (0, np.pi / 2.0))
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.grid(True)
        plt.colorbar(cmap_data)

        return ax

    def to_xarray(self):
        data = xr.DataArray(self._z,
                            dims=("phi", "theta"),
                            coords={"phi": self.azm, "theta": self.zen})
        return data


@attr.s
class PrincipalPlaneView(BRDFView):
    r"""Plots scattering in the principal plane, that is :math:`\phi = 0` and
    :math:`\phi = \pi`.

    Results from BRDF evaluation can be exported to a :class:`xarray.DataArray`
    holding two rows of values for the azimuth angles, instead of one row as
    the data are depicted in the plot.
    """

    def evaluate(self):
        self._z = np.zeros(np.shape(self.zen)[0] * 2)

        if isinstance(self.brdf, GriddedAdapter):
            thetas, phis, data = self.brdf._plotting_data(
                self.wi, self.wavelength)

            if 180 not in phis:
                raise ValueError(
                    "The principal plane view requires values at phi = 180°!")

            # Detect where the 0 and 180 azimuth are located
            phi_0deg = np.where(phis == 0)
            phi_180deg = np.where(phis == 180)

            # Initialise z plot values
            self.zen = np.deg2rad(thetas)
            n_thetas = len(thetas)
            self._z = np.zeros(n_thetas * 2)
            # FIXME: theta = 0 is duplicated for phi = 0 and phi = 180
            # Add a check and warn the user if it happens
            # Ignore one of the two values

            # Stitch both sides of the principal plane appropriately
            self._z[len(self.zen):] = data[:, phi_0deg].squeeze()
            self._z[:len(self.zen)] = data[::-1, phi_180deg].squeeze()

        elif isinstance(self.brdf, SampledAdapter):
            # phi=0 goes in the second half of the array
            for i in range(np.shape(self.zen)[0]):
                wo = [self.zen[i], 0]
                self._z[i + np.shape(self.zen)[0]] = \
                    self.brdf.evaluate(wo, self.wi, self.wavelength)

            # phi=pi goes in the first half of the array, but reversed
            for i in range(np.shape(self.zen)[0]):
                wo = [self.zen[i], np.pi]
                self._z[np.shape(self.zen)[0] - (i + 1)] = \
                    self.brdf.evaluate(wo, self.wi, self.wavelength)

        else:
            raise TypeError(f"Unsupported adapter {type(self.brdf)}!")

    def plot(self, ax=None):
        """
        Output the data to a 2D plot

        Parameter ``ax`` (:class:`~matplotlib.axes.Axes`)
            Axis object for attaching the plot
            If not set, the current axes object will be obtained from matplotlib

        Returns → :class:`~matplotlib.axes.Axes`
            An Axes object for use in custom matplotlib setups
        """
        if ax is None:
            ax = plt.gca()

        theta = np.concatenate((self.zen[::-1] * (-1), self.zen))
        ax.plot(theta, self._z, marker=".")
        ticks, labels = generate_ticks(11, [-np.pi / 2.0, np.pi / 2.0])
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        return ax

    def to_xarray(self):
        """The theta range is doubled for principal plane plots, yielding two entries
        with theta=0 in the middle of the array. Therefore the array is simply sliced
        in half for storage.

        Returns → :class:`xarray.DataArray`
            Data obtained from :meth:`evaluate`
        """
        halflength = len(self._z) / 2.0
        # since we double the theta array, theta=0 will show up twice
        if halflength % 1 == 0:
            data = xr.DataArray(
                np.array(
                    [self._z[int(halflength) - 1:: -1],
                     self._z[int(halflength):]]
                ),
                dims=("phi", "theta"),
                coords={"theta": self.zen, "phi": [np.pi, 0]},
            )
        else:
            data = xr.DataArray(
                np.array(
                    [
                        self._z[int(np.floor(halflength)):: -1],
                        self._z[int(np.floor(halflength)):],
                    ]
                ),
                dims=("phi", "theta"),
                coords={"theta": self.zen, "phi": [np.pi, 0]},
            )
        return data


def generate_ticks(num_ticks, limits):
    """Generates ticks and their respective tickmarks.

    Parameter ``num_ticks`` (int)
        Number of ticks to generate, including the limits
        of the given range
    Parameter ``limits`` (list[float])
        List of two values, limiting the ticks inclusive

    Returns → list, list
        - Values for the ticks
        - Tick values converted to degrees as string tickmarks
    """

    step_width = float(limits[1] - limits[0]) / (num_ticks - 1)

    steps = [limits[0] + step_width * i for i in range(num_ticks)]
    marks = [f"{i / np.pi * 180:.1f}°" for i in steps]

    return steps, marks
