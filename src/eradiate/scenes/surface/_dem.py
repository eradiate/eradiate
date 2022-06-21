from __future__ import annotations

import typing as t

import attr
import numpy as np
import pint
import xarray as xr

from ._core import Surface, surface_factory
from ..bsdfs import BSDF, LambertianBSDF, bsdf_factory
from ..core import KernelDict
from ..shapes import BufferMeshShape, FileMeshShape, Shape, shape_factory
from ...attrs import documented, get_doc, parse_docs
from ...contexts import KernelDictContext
from ...units import to_quantity
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg
from ...util.misc import onedict_value


@parse_docs
@attr.s
class DEMSurface(Surface):
    """
    DEM surface [``dem``]

    A surface description for digital elevation models. This object holds only the Shape.
    The associated BSDF is defined within the Shape plugin.

    This surface supports instantiation based on preprocessed geotiff files,
    triangulated meshes in the PLY and OBJ formats and analytical surfaces based on functions
    which map x and y coordinates to an elevation value.

    .. admonition:: Class method constructors

        .. autosummary::

            from_dataarray
            from_analytical

    Note: The DEMSurface overwrites :meth:`.Surface.kernel_dict`, because of the way, kernel dict
    creation is handled in :class:`BufferMeshShape`. That class returns a mitsuba Mesh object, instead
    of a kernel dict and so it cannot be updated to hold a reference to the BSDF specified here. Instead we
    overwrite the bsdf member of the underlying shape class directly before kernel dict creation.
    """

    id: t.Optional[str] = documented(
        attr.ib(
            default="terrain",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(Surface, "id", "doc"),
        type=get_doc(Surface, "id", "type"),
        init_type=get_doc(Surface, "id", "init_type"),
        default='"surface"',
    )

    shape: Shape = documented(
        attr.ib(
            converter=attr.converters.optional(shape_factory.convert),
            validator=attr.validators.optional(
                attr.validators.instance_of((BufferMeshShape, FileMeshShape))
            ),
            kw_only=True,
        ),
        doc="Shape describing the surface.",
        type=".BufferMeshShape or .OBJMeshShape or .PLYMeshShape",
        init_type=".BufferMeshShape or .OBJMeshShape or .PLYMeshShape or dict",
    )

    bsdf: BSDF = documented(
        attr.ib(
            factory=LambertianBSDF,
            converter=bsdf_factory.convert,
            validator=attr.validators.instance_of(BSDF),
        ),
        doc="The reflection model attached to the surface.",
        type=".BSDF",
        init_type=".BSDF or dict, optional",
        default=":class:`LambertianBSDF() <.LambertianBSDF>`",
    )

    @staticmethod
    def _vertex_index(x, y, len_y):
        return x * len_y + y

    @staticmethod
    def _generate_face_indices(len_x, len_y):
        face_indices = []
        for x in range(len_x + 1):
            for y in range(len_y + 1):
                vertex_sw = DEMSurface._vertex_index(x, y, len_y + 2)
                vertex_ne = DEMSurface._vertex_index(x + 1, y + 1, len_y + 2)
                vertex_nw = DEMSurface._vertex_index(x, y + 1, len_y + 2)
                vertex_se = DEMSurface._vertex_index(x + 1, y, len_y + 2)
                # In mitsuba, triangles are defined clockwise
                face_indices.append((vertex_sw, vertex_ne, vertex_nw))
                face_indices.append((vertex_sw, vertex_se, vertex_ne))

        return face_indices

    @classmethod
    def from_dataarray(
        cls, data: xr.DataArray, bsdf: BSDF, planet_radius=6371 * ureg.km
    ) -> DEMSurface:
        """
        This classmethod enables the instantiation of a DEM from an xarray data array holding elevation data.
        The dataarray must use latitude and longitude as its dimensional coordinates.

        Parameters
        ----------
        data : DataArray
            Dataarray holding the elevation data. Dimensional coordinates must be latitude `lat`
            and longitude `lon`. Data will be interpreted to be given in kernel units.

        bsdf : .BSDF
            BSDF to be attached to the mesh shape.

        planet_radius : quantity
            Planet radius used to convert latitude and longitude into distance units.
            Defaults to Earth's radius.

        Returns
        -------
        :class:`.DEMSurface`
            Created :class:`DEMSurface`.
        """
        radius = planet_radius.m_as(ucc.get("length"))

        len_lat = len(data.lat)  # x
        len_lon = len(data.lon)  # y

        # I convert the degrees to local distances based on the planet radius
        # This will be handled differently when I include the AtmosphereGeometry.
        lon_rad = to_quantity(data.lon).m_as(ureg.rad)
        lat_rad = to_quantity(data.lat).m_as(ureg.rad)
        mean_lon = lon_rad.mean()
        lon_range = lon_rad.max() - lon_rad.min()
        lat_range = lat_rad.max() - lat_rad.min()
        lon_length = (lon_range * radius)
        lat_length = (lat_range * radius * np.cos(mean_lon))

        lat_range = list(np.linspace(-lat_length / 2.0, lat_length / 2.0, len_lat))
        # Duplicate the first and last entry for the extra vertices, which form the
        # vertical walls
        lat_range.insert(0, lat_range[0])
        lat_range.append(lat_range[-1])

        lon_range = list(np.linspace(-lon_length / 2.0, lon_length / 2.0, len_lon))

        vertex_positions = np.zeros(((len_lat + 2) * (len_lon + 2), 3))
        values = np.array(data.data).transpose()[:, ::-1]
        for i, x in enumerate(lat_range):
            if i in [0, len_lat + 1]:
                # For the rows of vertices which are only the lower points of the wall
                # set the z coordinate to -1
                z_list = np.full(len_lon + 2, -1.0)
            else:
                z_list = values[i - 1]
                # Add z=-1 on both ends for the points that make up the vertical wall
                z_list = np.insert(z_list, 0, -1.0)
                z_list = np.append(z_list, -1.0)
            for j, z in enumerate(z_list):
                # Duplicate the first and last entry for the extra vertices, which form the
                # vertical walls
                if j == 0:
                    y = lon_range[0]
                elif j == len_lon + 1:
                    y = lon_range[-1]
                else:
                    y = lon_range[j - 1]
                vertex_positions[i * len_lon + j, :] = [x, y, z]

        # For each row and each column, except the last ones, define two
        # triangles, extending one index in each dimension.
        face_indices = cls._generate_face_indices(len_lat, len_lon)

        return cls(
            shape=BufferMeshShape(
                vertices=vertex_positions, faces=face_indices, bsdf=bsdf
            )
        )

    @classmethod
    def from_analytical(
        cls,
        elevation_function: t.Callable,
        x_length: pint.Quantity,
        x_steps: int,
        y_length: pint.Quantity,
        y_steps: int,
        bsdf: BSDF,
    ) -> DEMSurface:
        """
        This classmethod allows the creation of a DEM surface based on a function, which
        maps x,y points to an elevation.

        Parameters
        ----------
        elevation_function : callable
            Function that takes x, y points as inputs and returns the corresponding elevation
            value. Elevation values are interpreted in kernel units.

        x_length : pint.Quantity
            Extent of the mapped area along the x-axis.

        x_steps : int
            Number of data points to generate along the x-axis.

        y_length : pint.Quantity
            Extent of the mapped area along the y-axis.

        y_steps : int
            Number of data points to generate along the y-axis.

        bsdf : .BSDF
            BSDF to be attached to the mesh shape.
        """
        x_length_m = x_length.m_as(ucc.get("length"))
        y_length_m = y_length.m_as(ucc.get("length"))

        # duplicate the first and last entries in x and y, for the vertices
        # that form the vertical walls
        x_range = np.empty((x_steps + 2,))
        x_range[1:-1] = np.linspace(-x_length_m / 2.0, x_length_m / 2.0, x_steps)
        x_range[0] = x_range[1]
        x_range[-1] = x_range[-2]

        y_range = np.empty((y_steps + 2,))
        y_range[1:-1] = np.linspace(-y_length_m / 2.0, y_length_m / 2.0, y_steps)
        y_range[0] = y_range[1]
        y_range[-1] = y_range[-2]

        face_indices = cls._generate_face_indices(x_steps, y_steps)

        vertex_positions = []
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                if i in [0, x_steps + 1] or j in [0, y_steps + 1]:
                    z = -1.0
                else:
                    z = elevation_function(x, y)
                vertex_positions.append([x, y, z])

        return cls(
            shape=BufferMeshShape(
                vertices=vertex_positions, faces=face_indices, bsdf=bsdf
            )
        )

    def kernel_bsdfs(self, ctx: KernelDictContext) -> KernelDict:
        # Inherit docstring
        return self.bsdf.kernel_dict(ctx)

    def kernel_shapes(self, ctx: KernelDictContext) -> KernelDict:
        # Inherit docstring
        return self.shape.kernel_dict(ctx)

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        # Inherit docstring

        # This will overwrite any set BSDF
        self.shape.bsdf = self.bsdf

        # This is handled differently, because BufferMeshShape instantiates
        # a mitsuba object instead of returning a dict
        kernel_dict = {
            self.shape_id: onedict_value(self.kernel_shapes(ctx)),
        }

        return KernelDict(kernel_dict)
