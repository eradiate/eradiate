from __future__ import annotations

import typing as t
import warnings

import attrs
import numpy as np
import pint
import xarray as xr
from pinttr.util import ensure_units

from ._core import Surface
from ..bsdfs import BSDF, LambertianBSDF, bsdf_factory
from ..core import InstanceSceneElement, NodeSceneElement
from ..shapes import BufferMeshShape, FileMeshShape, Shape, shape_factory
from ...attrs import documented, get_doc, parse_docs
from ...units import to_quantity
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg


@parse_docs
@attrs.define(eq=False, slots=False)
class DEMSurface(Surface):
    """
    DEM surface [``dem``]

    A surface description for digital elevation models.

    This surface supports instantiation based on an xarray data array, a
    triangulated mesh file in the PLY and OBJ formats and an analytical
    elevation mapping the x and y coordinates to an elevation value.

    .. admonition:: Class method constructors

       .. autosummary::

          from_dataarray
          from_analytical

    Notes
    -----
    * Contrary to most other surfaces, this scene element expands as a single
      Mitsuba Shape plugin which includes its child BSDF rather than referencing
      a top-level dictionary entry. The reason for this is that some allowed
      shapes expand as Mitsuba instances rather than dictionaries, which
      prevents object referencing.
    """

    id: str | None = documented(
        attrs.field(
            default="terrain",
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc=get_doc(Surface, "id", "doc"),
        type=get_doc(Surface, "id", "type"),
        init_type=get_doc(Surface, "id", "init_type"),
        default='"surface"',
    )

    shape: Shape = documented(
        attrs.field(
            converter=attrs.converters.optional(shape_factory.convert),
            validator=attrs.validators.optional(
                attrs.validators.instance_of((BufferMeshShape, FileMeshShape))
            ),
            kw_only=True,
        ),
        doc="Shape describing the surface.",
        type=".BufferMeshShape or .FileMeshShape",
        init_type=".BufferMeshShape or .OBJMeshShape or .PLYMeshShape or dict",
    )

    bsdf: BSDF = documented(
        attrs.field(
            factory=LambertianBSDF,
            converter=bsdf_factory.convert,
            validator=attrs.validators.instance_of(BSDF),
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
        cls,
        data: xr.DataArray,
        bsdf: BSDF,
        planet_radius: pint.Quantity | float = 6371.0 * ureg.km,
    ) -> DEMSurface:
        """
        Construct a DEM from an xarray data array holding elevation data.
        The data array must use latitude and longitude as its dimensional
        coordinates.

        Parameters
        ----------
        data : DataArray
            Data array holding the elevation data. Dimensional coordinates must
            be latitude `lat` and longitude `lon`. Data will be interpreted to
            be given in kernel units.

        bsdf : .BSDF
            BSDF to be attached to the mesh shape.

        planet_radius : quantity or float, optional
            Planet radius used to convert latitude and longitude into distance
            units. Unitless values are interpreted in default configuration
            units. Defaults to Earth's radius.

        Returns
        -------
        :class:`.DEMSurface`
        """
        radius = ensure_units(planet_radius, ucc.get("length"), convert=True).magnitude

        len_lat = len(data.lat)  # x
        len_lon = len(data.lon)  # y

        # Convert the degrees to local distances based on planet radius
        # TODO: Update this when adding support for spherical shell geometry
        lon_rad = to_quantity(data.lon).m_as(ureg.rad)
        lat_rad = to_quantity(data.lat).m_as(ureg.rad)
        mean_lon = lon_rad.mean()
        lon_range = lon_rad.max() - lon_rad.min()
        lat_range = lat_rad.max() - lat_rad.min()
        lon_length = lon_range * radius
        lat_length = lat_range * radius * np.cos(mean_lon)

        lat_range = list(np.linspace(-lat_length / 2.0, lat_length / 2.0, len_lat))
        # Duplicate the first and last entry for the extra vertices, which form
        # the vertical walls
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
            shape=BufferMeshShape(vertices=vertex_positions, faces=face_indices),
            bsdf=bsdf,
        )

    @classmethod
    def from_analytical(
        cls,
        elevation_function: t.Callable,
        x_length: pint.Quantity | float,
        x_steps: int,
        y_length: pint.Quantity | float,
        y_steps: int,
        bsdf: BSDF,
    ) -> DEMSurface:
        """
        Construct a DEM from an analytical function mapping the x and y
        coordinates to elevation values.

        Parameters
        ----------
        elevation_function : callable
            Function that takes x, y points as inputs and returns the
            corresponding elevation value. Elevation values are interpreted in
            kernel units.

        x_length : pint.Quantity or float
            Extent of the mapped area along the x-axis. Unitless values are
            interpreted in default configuration units.

        x_steps : int
            Number of data points to generate along the x-axis.

        y_length : pint.Quantity
            Extent of the mapped area along the y-axis. Unitless values are
            interpreted in default configuration units.

        y_steps : int
            Number of data points to generate along the y-axis.

        bsdf : .BSDF
            BSDF to be attached to the mesh shape.
        """
        x_length_m = ensure_units(x_length, ucc.get("length"), convert=True).magnitude
        y_length_m = ensure_units(y_length, ucc.get("length"), convert=True).magnitude

        # Duplicate the first and last entries in x and y, for the vertices
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
            shape=BufferMeshShape(vertices=vertex_positions, faces=face_indices),
            bsdf=bsdf,
        )

    def update(self) -> None:
        # Fix BSDF ID
        self.bsdf.id = self._bsdf_id

        # Fix shape ID
        self.shape.id = self._shape_id

        # Force BSDF nesting if the shape is defined
        if self.shape is not None:
            if isinstance(self.shape.bsdf, BSDF):
                warnings.warn("Set BSDF will be overridden by surface BSDF settings.")
            self.shape.bsdf = self.bsdf

    @property
    def _shape_id(self):
        """
        Mitsuba shape object identifier.
        """
        return f"{self.id}_shape"

    @property
    def _bsdf_id(self):
        """
        Mitsuba BSDF object identifier.
        """
        return f"{self.id}_bsdf"

    @property
    def _template_bsdfs(self) -> dict:
        return {}

    @property
    def _template_shapes(self) -> dict:
        return {}

    @property
    def _params_bsdfs(self) -> dict:
        return {}

    @property
    def _params_shapes(self) -> dict:
        return {}

    @property
    def objects(self) -> dict[str, NodeSceneElement | InstanceSceneElement]:
        # Inherit docstring
        return {self._shape_id: self.shape}
