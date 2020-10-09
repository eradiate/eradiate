""" Heterogeneous atmosphere scene generation helpers """
from pathlib import Path

import attr
import numpy as np
import xarray as xr

from .base import Atmosphere
from ..core import SceneHelperFactory
from ...util.attrs import attrib
from ...util.units import kernel_default_units as kdu


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


@SceneHelperFactory.register("heterogeneous")
@attr.s
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

    Constructor arguments / instance attributes:
        ``sigma_t`` (path-like):
            Path to the extinction coefficient volume data file.

            *Required* (no default).

        ``albedo`` (path-like):
            Path to the single scattering albedo volume data file.

            *Required* (no default).
    """

    albedo = attrib(
        default=None,
        converter=Path,
        validator=attr.validators.instance_of(Path)
    )

    sigma_t = attrib(
        default=None,
        converter=Path,
        validator=attr.validators.instance_of(Path)
    )

    @property
    def _width(self):
        """Return scene width based on configuration."""

        if self.width == "auto":  # Support for auto is currently disabled
            raise NotImplementedError
        else:
            return self.get_quantity("width")

    def phase(self):
        return {f"phase_{self.id}": {"type": "rayleigh"}}

    def media(self, ref=False):
        from eradiate.kernel.core import ScalarTransform4f

        width = self._width.to(kdu.get("length")).magnitude
        height, offset = self._height
        height = height.to(kdu.get("length")).magnitude
        offset = offset.to(kdu.get("length")).magnitude

        # First, transform the [0, 1]^3 cube to the right dimensions
        trafo = ScalarTransform4f([
            [width, 0., 0., -0.5 * width],
            [0., width, 0., -0.5 * width],
            [0., 0., height + offset, -offset],
            [0., 0., 0., 1.],
        ])

        # Output kernel dict
        return {
            f"medium_{self.id}": {
                "type": "heterogeneous",
                "phase": {"type": "rayleigh"},
                "sigma_t": {
                    "type": "gridvolume",
                    "filename": str(self.sigma_t),
                    "to_world": trafo
                },
                "albedo": {
                    "type": "gridvolume",
                    "filename": str(self.albedo),
                    "to_world": trafo
                },
            }
        }

    def shapes(self, ref=False):
        from eradiate.kernel.core import ScalarTransform4f

        if ref:
            medium = {"type": "ref", "id": f"medium_{self.id}"}
        else:
            medium = self.media(ref=False)[f"medium_{self.id}"]

        width = self._width.to(kdu.get("length")).magnitude
        height, offset = self._height
        height = height.to(kdu.get("length")).magnitude
        offset = offset.to(kdu.get("length")).magnitude

        return {
            f"shape_{self.id}": {
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
