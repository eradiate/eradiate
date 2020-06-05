from abc import ABC, abstractmethod

import attr
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import eradiate.kernel
from ..util import frame


@attr.s
class BRDFAdapter(ABC):
    """
    Wrapper class for different B[RS]DF backends.
    """

    @abstractmethod
    def evaluate(self, wo, wi, wavelength):
        """
        Evaluate the BRDF in a way appropriate for the implementation
        """
        pass


@attr.s
class MitsubaBSDFPluginAdapter(BRDFAdapter):
    r"""Wrapper class around eradiate.kernel.BSDF plugins.
    Holds a BSDF plugin and exposes an evaluate method, taking care of internal
    necessities for evaluating the plugin.

    Instance attributes:
        ``bsdf`` (:class:`eradiate.kernel.render.BSDF`): Mitsuba BSDF plugin
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
        si.wi = wi
        si.wavelengths = [wavelength]
        if len(wo) == 2:
            wo = frame.angles_to_direction(wo[0], wo[1])

        return self.bsdf.eval(ctx, si, wo)[0] / wo[2]

    def sample(self):
        raise NotImplementedError("Coming soon.")


@attr.s
class DataArrayBRDFAdapter(BRDFAdapter):
    r"""Wrapper class around xarray formatted data from Eradiate solvers.
    Holds gridded data obtained through radiative transfer computation and exposes
    an evaluate method. Gridded data is interpolated linearly between data points
    and clipped to 0 outside of the data boundaries.

    If data along the :math:`\phi_o` axis does not contain data for 0° and 360° one value
    is copied to the other position to complete the data set for interpolation.

    This class expects the xarray data to be formatted in the following way:

    - The array has 5 dimensions, named and ordered like this:
        :code:`['theta_i', 'phi_i', 'theta_o', 'phi_o', 'wavelength']`
    - The phi_o dimension must contain values for at least one of the two values
      0° and 360° for interpolation. If only one is present, it is used for
      interpolation from both directions

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
        if 0 not in self.data.phi_o:
            self.data = self.copy_phi_o_values(360, 0)
        elif 360 not in self.data.phi_o:
            self.data = self.copy_phi_o_values(0, 360)
        elif 0 not in self.data.phi_o and 360 not in self.data.phi_o:
            raise ValueError("Data contains data for neither 0° nor 360°!")

    def _copy_phi_o_values(self, origin, target):
        r"""Constructs a new xarray, containing one extra value in the ":math:`phi_o`" dimension.
        Data from :math:`phi_o` == origin is copied into :math:`phi_o` == target.
        """

        coords = {dim: self.data.coords[dim].values for dim in self.data.dims}
        coords["phi_o"] = np.append(coords["phi_o"], target)

        empty = np.zeros([len(coords[dim])
                          for dim in self.data.dims])
        data_new = xr.DataArray(empty, [coords[dim]
                                        for dim in self.data.dims], self.data.dims)

        theta_i_ind = xr.DataArray(
            np.arange(len(self.data.coords["theta_i"])), dims=["theta_i"])
        phi_i_ind = xr.DataArray(
            np.arange(len(self.data.coords["phi_i"])), dims=["phi_i"])
        theta_o_ind = xr.DataArray(
            np.arange(len(self.data.coords["theta_o"])), dims=["theta_o"])
        phi_o_ind = xr.DataArray(
            np.arange(len(self.data.coords["phi_o"])), dims=["phi_o"])
        wavelength_ind = xr.DataArray(
            np.arange(len(self.data.coords["wavelength"])), dims=["wavelength"])

        data_new[theta_i_ind, phi_i_ind, theta_o_ind, phi_o_ind,
                 wavelength_ind] = self.data[theta_i_ind, phi_i_ind, theta_o_ind, phi_o_ind, wavelength_ind]
        data_new = xr.where(
            (data_new.coords["phi_o"] == target), self.data.sel(phi_o=origin), data_new)

        return data_new

    def _split_dimensions_for_interp(self, **kwargs):
        sel_kwargs = {}
        interp_kwargs = {}
        for dim, coords in kwargs.items():
            if len(self.data.coords[dim]) == 1:
                sel_kwargs[dim] = coords
            else:
                interp_kwargs[dim] = coords
        return sel_kwargs, interp_kwargs

    def evaluate(self, wo, wi, wavelength):
        if len(wi) == 3:
            wi = frame.direction_to_angles(wi)
        if len(wo) == 3:
            wo = frame.direction_to_angles(wo)

        selkws, interpkws = self._split_dimensions_for_interp(
            theta_i=np.rad2deg(wi[0]), phi_i=np.rad2deg(wi[1]),
            theta_o=np.rad2deg(wo[0]), phi_o=np.rad2deg(wo[1]), wavelength=wavelength)
        print("selkws:", selkws)
        print("interpkws:", interpkws)

        try:
            return self.data.sel(**selkws).interp(**interpkws, kwargs={"fill_value": 0.0}).data
        except KeyError as e:
            e.message = e.message + \
                "\nNon interpolateable dimension was called with a value that does not fit the data points!"
            raise e


@attr.s
class BRDFView(ABC):
    """The BRDFViewer class is a utility to visualize bidirectional reflection distribution
    functions. It can read data from Mitsuba plugins as well as BRDF files computed
    with Eradiate itself.

    It can create polar plots as well as linear plots of the principal plane.

    Instance attributes:
        ``brdf`` (:class:`BRDF`):
            Source of scattering data to be evaluated and plotted
        ``wi`` (:class:`float`):
            Incoming direction of radiation, can be set in cartesian
            or spherical (in degrees) coordinates
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
    _wi = attr.ib(default=[0.0, 0.0, 1.0])
    _azm = attr.ib(default=np.linspace(0, np.pi * 2, 101))
    _zen = attr.ib(default=np.linspace(0, np.pi / 2.0, 101))
    _wavelength = attr.ib(default=650)
    _z = attr.ib(init=False)

    @property
    def wavelength(self):
        """Wavelength in nanometers to evaluate the BSDF at.
        For BSDFs that are specified without spectral dependency, this value
        can be left at its default value of 650nm.
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
            self._wi = frame.angles_to_direction(
                np.deg2rad(wi[0]), np.deg2rad(wi[1]))
        elif len(wi) == 3:
            norm = np.linalg.norm(wi)
            if norm == 0:
                raise ValueError(
                    "Incoming direction vector cannot have length 0!")
            else:
                self._wi = wi / np.linalg.norm(wi)

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
            self._brdf = DataArrayBRDFAdapter(source)
        elif isinstance(source, str):
            raise TypeError("Loading BRDF data files is not supported yet.")
        else:
            raise TypeError(f"Non supported BRDF source type: {type(source)}")

    @abstractmethod
    def evaluate(self):
        """Evaluate the BRDF object, filling the data structures necessary for
        the desired plot.
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


class PolarView(BRDFView):
    """Creates a polar plot of scattering into the hemisphere which holds the incoming
    radiation and the surface normal.
    """

    def evaluate(self):
        self.r, self.th = np.meshgrid(self.zen, self.azm)
        self._z = np.zeros(np.shape(self.r))

        for i in range(np.shape(self.r)[0]):
            for j in range(np.shape(self.r)[1]):
                wo = frame.angles_to_direction(self.r[i][j], self.th[i][j])
                self._z[i][j] = self.brdf.evaluate(
                    wo, self.wi, self.wavelength)

    def plot(self, ax=None):
        """
        Output the data to a polar contour plot

        Parameter ``ax`` (Axes):
            Axis object for attaching the plot
            If not set, the current axes object will be obtained from matplotlib

        Returns → Axes:
            An Axes object for use in custom matplotlib setups
        """
        if ax is None:
            ax = plt.gca(projection="polar")

        contour = ax.contourf(self.th, self.r, self._z, cmap="BuPu")
        ticks, labels = generate_ticks(5, (0, np.pi / 2.0))
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.grid(True)
        plt.colorbar(contour)

        return ax

    def to_xarray(self):
        data = xr.DataArray(
            self._z, dims=("phi", "theta"), coords={"phi": self.azm, "theta": self.zen}
        )
        return data


@attr.s
class PrincipalPlaneView(BRDFView):
    r"""Plots scattering in the principal plane, that is :math:`\phi=0` and :math:`\phi=\pi`

    Results from BRDF evaluation can be exported to an xarray, holding two rows
    of values, for the azimuth angles, instead of one row, as the data are depicted
    in the plot.
    """

    def evaluate(self):
        self._z = np.zeros(np.shape(self.zen)[0] * 2)
        # phi=0 goes in the second half of the array
        for i in range(np.shape(self.zen)[0]):
            wo = frame.angles_to_direction(self.zen[i], 0)
            self._z[i + np.shape(self.zen)[0]] = self.brdf.evaluate(
                wo, self.wi, self.wavelength
            )

        # phi=pi goes in the first half of the array, but reversed
        for i in range(np.shape(self.zen)[0]):
            wo = frame.angles_to_direction(self.zen[i], np.pi)
            self._z[np.shape(self.zen)[0] - (i + 1)] = self.brdf.evaluate(
                wo, self.wi, self.wavelength
            )

    def plot(self, ax=None):
        """
        Output the data to a 2D plot

        Parameter ``ax`` (Axes)
            Axis object for attaching the plot
            If not set, the current axes object will be obtained from matplotlib

        Returns → Axes
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

        Returns → xarray.DataArray
            Data obtained from evaluate
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
