""" Heterogeneous atmosphere scene generation helpers """

import attr
import numpy as np
import xarray as xr

from .base import Atmosphere
from ..core import Factory
from ...util.units import kernel_default_units as kdu, config_default_units as cdu


def write_binary_grid3d(filename, values):
    """Writes volume data to a binary file so that a ``gridvolume`` kernel
    plugin can be instantiated with that file.

    Parameter ``filename`` (path-like):
        File name.

    Parameter ``values`` (:class:`~numpy.ndarray` or :class:`~xarray.DataArray`):
        Data array to output to the volume data file. This array must 3 or 4
        dimensions (x, y, z, spectrum). If the array is 3-dimensional, it will
        automatically be assumed to have only one spectral channel.
    """
    if isinstance(values, xr.DataArray):
        values = values.values

    if not isinstance(values, np.ndarray):
        raise TypeError(f"unsupported data type {type(values)} "
                        f"(expected numpy array or xarray DataArray)")

    # note: this is an exact copy of the function write_binary_grid3d from
    # https://github.com/mitsuba-renderer/mitsuba-data/blob/master/tests/scenes/participating_media/create_volume_data.py

    with open(filename, 'wb') as f:
        f.write(b'V')
        f.write(b'O')
        f.write(b'L')
        f.write(np.uint8(3).tobytes())  # Version
        f.write(np.int32(1).tobytes())  # type
        f.write(np.int32(values.shape[0]).tobytes())  # size
        f.write(np.int32(values.shape[1]).tobytes())
        f.write(np.int32(values.shape[2]).tobytes())
        if values.ndim == 3:
            f.write(np.int32(1).tobytes())  # channels
        else:
            f.write(np.int32(values.shape[3]).tobytes())  # channels
        f.write(np.float32(0.0).tobytes())  # bbox
        f.write(np.float32(0.0).tobytes())
        f.write(np.float32(0.0).tobytes())
        f.write(np.float32(1.0).tobytes())
        f.write(np.float32(1.0).tobytes())
        f.write(np.float32(1.0).tobytes())
        f.write(values.ravel().astype(np.float32).tobytes())


@attr.s
@Factory.register("heterogeneous")
class HeterogeneousAtmosphere(Atmosphere):
    r"""Heterogeneous atmosphere scene generation helper
    [:factorykey:`heterogeneous`].

    This class builds a one-dimensional heterogeneous atmosphere. The used
    optical properties are specified as binary files. Binary data files can
    be generated from :class:`~xarray.DataArray` s or :class:`~numpy.ndarray` s
    using the :func:`write_binary_grid3d` function.

    .. warning::

       Optical property data will not be scaled by default unit override:
       they must manually be specified in the appropriate kernel units.

    .. admonition:: Configuration example
        :class: hint

        Default:
            .. code:: python

               {
                   "height": 100.,
                   "width": "auto",
                   "sigma_t": "sigma_t.vol",
                   "albedo": "albedo.vol",
               }

    .. admonition:: Configuration format
        :class: hint

        ``height`` (float):
            Height of the atmosphere [km].

            Default: 100.

        ``width`` (float)
            Width of the atmosphere [km].

            Default: 100.

        ``sigma_t`` (string):
            Path to the extinction coefficient volume data file.

            *Required* (no default).

        ``albedo`` (string):
            Path to the single scattering albedo volume data file.

            *Required* (no default).
    """

    def config_schema(cls):
        return {
            "height": {
                "type": "number",
                "min": 0.,
                "default": 100.,
            },
            "height_unit": {
                "type": "string",
                "default": cdu.get_str("length")
            },
            "width": {
                "anyof": [{
                    "type": "number",
                    "min": 0.
                }, {
                    "type": "string",
                    "allowed": ["auto"]
                }],
                "default": 100.
            },
            "width_unit": {
                "type": "string",
                "required": False,
                "nullable": True,
                "default_setter": lambda doc:
                None if isinstance(doc["width"], str)
                else cdu.get_str("length")
            },
            "sigma_t": {
                "type": "string",
                "required": True,
            },
            "albedo": {
                "type": "string",
                "required": True,
            }
        }

    @property
    def _albedo(self):
        return self.config["albedo"]

    @property
    def _height(self):
        height = self.config.get_quantity("height").to(kdu.get("length")).magnitude
        offset = height * 0.001  # TODO: maybe adjust offset based on medium profile
        return height, offset

    @property
    def _sigma_t(self):
        return self.config["sigma_t"]

    @property
    def _width(self):
        """Return scene width based on configuration."""
        width = self.config.get_quantity("width")

        if width == "auto":  # Support for auto is currently disabled
            raise NotImplementedError
        else:
            return width.to(kdu.get("length")).magnitude

    def phase(self):
        return {"phase_atmosphere": {"type": "rayleigh"}}

    def media(self, ref=False):
        from eradiate.kernel.core import ScalarTransform4f

        width = self._width
        height, offset = self._height

        # First, transform the [0, 1]^3 cube to the right dimensions
        trafo = ScalarTransform4f([
            [width, 0., 0., -0.5 * width],
            [0., width, 0., -0.5 * width],
            [0., 0., height + offset, -offset],
            [0., 0., 0., 1.],
        ])

        # Output kernel dict
        return {
            "medium_atmosphere": {
                "type": "heterogeneous",
                "phase": {"type": "rayleigh"},
                "sigma_t": {
                    "type": "gridvolume",
                    "filename": self._sigma_t,
                    "to_world": trafo
                },
                "albedo": {
                    "type": "gridvolume",
                    "filename": self._albedo,
                    "to_world": trafo
                },
            }
        }

    def shapes(self, ref=False):
        from eradiate.kernel.core import ScalarTransform4f

        if ref:
            medium = {"type": "ref", "id": "medium_atmosphere"}
        else:
            medium = self.media(ref=False)["medium_atmosphere"]

        width = self._width
        height, offset = self._height

        return {
            "shape_atmosphere": {
                "type":
                    "cube",
                "to_world":
                    ScalarTransform4f([
                        [0.5 * width, 0., 0., 0.],
                        [0., 0.5 * width, 0., 0.],
                        [0., 0., 0.5 * (height + offset), 0.5 * (height - offset)],
                        [0., 0., 0., 1.],
                    ]),
                "bsdf": {
                    "type": "null"
                },
                "interior":
                    medium
            }
        }
