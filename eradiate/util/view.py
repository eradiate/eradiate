"""Legacy code, currently unmaintained."""

# TODO: finish salvaging the code which should be kept and remove this module.

from abc import ABC, abstractmethod

import attr
import numpy as np

import eradiate.kernel
from .xarray import make_dataarray
from ..util import frame


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
    arr = make_dataarray(data, [sza], [saa], vza, vaa, wavelength, "hemispherical")

    return arr
