from __future__ import annotations

import warnings

import attrs
import mitsuba as mi
import numpy as np
import pint
import xarray as xr

from ._core import Surface
from ..bsdfs import BSDF, LambertianBSDF, OpacityMaskBSDF
from ..core import SceneElement
from ..geometry import PlaneParallelGeometry, SceneGeometry, SphericalShellGeometry
from ..shapes import (
    BufferMeshShape,
    FileMeshShape,
    RectangleShape,
    Shape,
    SphereShape,
    shape_factory,
)
from ...attrs import documented, get_doc, parse_docs
from ...constants import EARTH_RADIUS
from ...units import to_quantity
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


def mesh_from_dem(
    da: xr.DataArray,
    geometry: str | dict | SceneGeometry,
    planet_radius: pint.Quantity | float | None = None,
) -> "mitsuba.Mesh":
    """
    Construct a DEM surface from a data array holding elevation data.

    Parameters
    ----------
    da : DataArray
        Data array with elevation data, indexed either by latitude and longitude
        coordinates or x and y coordinates.

    geometry : .SceneGeometry or dict or str
        Scene geometry configuration. The value is pre-processed by the
        :meth:`.SceneGeometry.convert` converter.

    planet_radius : quantity or float, default: .EARTH_RADIUS
        Planet radius used to convert latitude/longitude to x/y when
        `geometry` is a :class:`.PlaneParallelGeometry` instance.
        This parameter is unused otherwise. If a unitless value is passed, it is
        interpreted using
        :ref:`default config length units <sec-user_guide-unit_guide_user>`.

    Returns
    -------
    mesh : .BufferMeshShape
        A triangulated mesh representing the DEM.

    theta_lim : pint.Quantity
        The limits of the latitude extent of the DEM.

    phi_lim : pint.Quantity
        The limits of the longitude extent of the DEM.

    Notes
    -----
    The ``da`` parameter may use the following formats:

    * with latitude and longitude based coordinates, then named ``"lat"`` and
      ``"lon"``;
    * with x and y length based coordinates, then named ``"x"`` and ``"y"``.

    Coordinate and variable units are specified using the ``units`` xarray
    attributes.
    """
    # Pre-process geometry parameter
    geometry = SceneGeometry.convert(geometry)

    if isinstance(geometry, SphericalShellGeometry) and planet_radius is not None:
        warnings.warn("SphericalShellGeometry overrides the `planet_radius` argument.")

    # Add default units if quantity is unitless
    if planet_radius is not None and not isinstance(planet_radius, pint.Quantity):
        planet_radius = planet_radius * ucc.get("length")
    # Set default if in a plane-parallel setup
    planet_radius = EARTH_RADIUS if planet_radius is None else planet_radius

    if "lat" in da.coords and "lon" in da.coords:
        elevation = to_quantity(da.transpose("lat", "lon"))
        lat = ureg.Quantity(da.lat.values, da["lat"].attrs["units"]).m_as(ureg.deg)
        lon = ureg.Quantity(da.lon.values, da["lon"].attrs["units"]).m_as(ureg.deg)
        vertices = _generate_dem_vertices(lat, lon, elevation.m_as(uck.get("length")))
        faces = _generate_face_indices(len(lat), len(lon))

    elif "x" in da.coords and "y" in da.coords:
        elevation = to_quantity(da.transpose("x", "y"))
        if isinstance(geometry, SphericalShellGeometry):
            planet_radius = geometry.planet_radius

        x = np.rad2deg(
            (ureg.Quantity(da.x.values, da["x"].attrs["units"]) / planet_radius).m
        )
        y = np.rad2deg(
            (ureg.Quantity(da.y.values, da["y"].attrs["units"]) / planet_radius).m
        )

        vertices = _generate_dem_vertices(x, y, elevation.m_as(uck.get("length")))
        faces = _generate_face_indices(len(x), len(y))

    else:
        raise ValueError(
            f"Data array coordinates must be either `x/y` or `lat/lon`.\nGot: {da.coords}"
        )

    theta_lim, phi_lim = _minmax_coordinates(vertices)
    theta_mean, phi_mean = _mean_coordinates(vertices)
    atmo_bottom = geometry.ground_altitude

    if isinstance(geometry, SphericalShellGeometry):
        trafo = (
            mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=-90)
            @ mi.ScalarTransform4f.rotate(axis=[0, 1, 0], angle=-(90 - theta_mean))
            @ mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=-phi_mean)
        ).matrix.numpy()[:3, :3]

        vertices_spherical = _transform_vertices_spherical_shell(
            vertices, geometry.planet_radius.m_as(uck.get("length"))
        )
        vertices_new = [trafo.dot(vertex) for vertex in vertices_spherical] * uck.get(
            "length"
        )

        mesh = BufferMeshShape(vertices=vertices_new, faces=faces)

    elif isinstance(geometry, PlaneParallelGeometry):
        vertices_new = _transform_vertices_plane_parallel(
            vertices,
            planet_radius=planet_radius.m_as(uck.get("length")),
            altitude=atmo_bottom.m_as(uck.get("length")),
        ) * uck.get("length")
        mesh = BufferMeshShape(vertices=vertices_new, faces=faces)

    else:
        raise ValueError(f"Unsupported scene geometry: {geometry}")

    return mesh, theta_lim * ureg.deg, phi_lim * ureg.deg


def _generate_dem_vertices(x, y, elevation):
    """
    Generate DEM vertex positions as (latitude, longitude, elevation) triplets
    for a plane parallel geometry.

    Parameters
    ----------
    x : array-like
        Points along the latitude axis of the DEM

    y : array-like
        Points along the longitude axis of the DEM

    elevation : array-like
        Elevation data

    Returns
    -------
    vertices : array
        DEM vertices as (lat, lon, elevation) triplets.
    """
    y_gr, x_gr = np.meshgrid(y, x)

    vertices = np.array((x_gr.ravel(), y_gr.ravel(), elevation.ravel())).transpose()
    return vertices


def _vertex_index(x, y, len_y):
    """
    Vertex index for a given row and column of gridded vertex data
    This function handles vectorized index lookup with numpy arrays,
    as well as individual lookup using floats.

    Parameters
    ----------
    x : np.array
        Row index of the vertex grid
    y : np.array
        Column index of the vertex grid
    len_y : float
        Length of the second dimension of the vertex grid

    Returns
    -------
    index : np.array
        Index of the vertex in a flattened structure.
    """
    return x * len_y + y


def _generate_face_indices(len_x, len_y):
    """
    Generate indices for face definition in a mesh with gridded vertex positions

    Parameters
    ----------
    len_x : float
        Length of the row dimension of the vertex grid

    len_y : float
        Length of the column dimension of the vertex grid

    Returns
    -------
    face_indices : np.array
        (n, 3) array defining faces of the triangulated mesh for the DEM.
    """
    x = np.array(range(len_x - 1))
    y = np.array(range(len_y - 1))
    xg, yg = np.meshgrid(x, y)
    vertex_sw = _vertex_index(xg.flatten(), yg.flatten(), len_y)

    xg, yg = np.meshgrid(x + 1, y + 1)
    vertex_ne = _vertex_index(xg.flatten(), yg.flatten(), len_y)

    xg, yg = np.meshgrid(x, y + 1)
    vertex_nw = _vertex_index(xg.flatten(), yg.flatten(), len_y)

    xg, yg = np.meshgrid(x + 1, y)
    vertex_se = _vertex_index(xg.flatten(), yg.flatten(), len_y)

    face_indices_1 = np.array((vertex_ne, vertex_sw, vertex_nw)).transpose()
    face_indices_2 = np.array((vertex_se, vertex_sw, vertex_ne)).transpose()
    face_indices = np.concatenate((face_indices_1, face_indices_2))

    return face_indices


def _transform_vertices_spherical_shell(vertices, planet_radius):
    """
    Convert the (lat, lon, elevation) vertices from the initial vertex generation
    into (x, y, z) values for spherical shell geometries.

    Parameters
    ----------
    vertices : array-like
        List of mesh vertices in (lat, lon, elevation) tuples.

    planet_radius : float
        Planet radius; assumed to be given in kernel units
    """
    theta_r = np.deg2rad(90 - vertices[:, 0])
    phi_r = np.deg2rad(vertices[:, 1])
    x = np.sin(theta_r) * np.cos(phi_r) * (planet_radius + vertices[:, 2])
    y = np.sin(theta_r) * np.sin(phi_r) * (planet_radius + vertices[:, 2])
    z = np.cos(theta_r) * (planet_radius + vertices[:, 2])
    vertices_new = np.array((x, y, z)).transpose()

    return vertices_new


def _transform_vertices_plane_parallel(vertices, planet_radius, altitude):
    """
    Convert the (lat, lon, elevation) vertices from the initial vertex generation
    into (x, y, z) values for plane parallel atmosphere geometries.

    Parameters
    ----------
    vertices : array-like
        List of mesh vertices in (lat, lon, elevation) tuples.

    planet_radius : float
        Planet radius; assumed to be given in kernel units

    altitude : float
        Atmosphere bottom altitude; assumed to be given in kernel units
    """
    theta_mean, phi_mean = _mean_coordinates(vertices)
    phi_local = np.deg2rad(vertices[:, 1] - phi_mean)
    theta_local = np.deg2rad(vertices[:, 0] - theta_mean)

    x = phi_local * planet_radius
    y = theta_local * planet_radius
    z = vertices[:, 2] + altitude

    vertices_new = np.array((x, y, z)).transpose()
    return vertices_new


def _mean_coordinates(vertices):
    """
    Return the mean of the latitude and longitude values in the vertex list.
    The mean is computed as the average of the smallest and largest value in each set.
    """
    (theta_min, theta_max), (phi_min, phi_max) = _minmax_coordinates(vertices)

    phi_mean = ((phi_max + phi_min) / 2.0) % 360
    theta_mean = (theta_max + theta_min) / 2.0
    return theta_mean, phi_mean


def _minmax_coordinates(vertices):
    """
    Return the minimum and maximum values of the latitude and longitude values
    in the vertex list.
    """
    theta_min = np.min(vertices[:, 0])
    theta_max = np.max(vertices[:, 0])
    phi_min = np.min(vertices[:, 1])
    phi_max = np.max(vertices[:, 1])
    return (theta_min, theta_max), (phi_min, phi_max)


def _to_uv(lat_min, lat_max, lon_min, lon_max) -> "mitsuba.ScalarTransform4f":
    """
    Compute the `to_uv` transformation for the opacity mask bitmap. To do this,
    the latitude and longitude extent of the DEM specification are used.

    To avoid intersection between the terrain mesh and the surrounding background
    shape, an opacity mask is attached to the background's BSDF.

    """
    lon_range = lon_max - lon_min
    lon_scale = 120 / lon_range

    lat_range = lat_max - lat_min
    lat_scale = 60 / lat_range

    lon_mean = (lon_max + lon_min) / 2.0
    lon_uv = lon_mean / 360 + 0.5

    lat_mean = (lat_max + lat_min) / 2.0
    lat_uv = 0.5 - (lat_mean / 180)

    return mi.ScalarTransform4f.scale(
        (lon_scale, lat_scale, 1)
    ) @ mi.ScalarTransform4f.translate(
        (-lon_uv + (0.5 / lon_scale), -lat_uv + (0.5 / lat_scale), 0)
    )


@parse_docs
@attrs.define(eq=False, slots=False)
class DEMSurface(Surface):
    """
    DEM Surface [``dem``]

    A mesh based representation of surface DEMs.
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
            default=None,
        ),
        doc="Shape describing the surface.",
        type=".BufferMeshShape or .FileMeshShape or None",
        init_type=".BufferMeshShape or .OBJMeshShape or .PLYMeshShape or dict",
        default="None",
    )

    shape_background: Shape = documented(
        attrs.field(
            converter=attrs.converters.optional(shape_factory.convert),
            validator=attrs.validators.optional(
                attrs.validators.instance_of((SphereShape, RectangleShape))
            ),
            kw_only=True,
            default=None,
        ),
        doc="Shape describing the background surface.",
        type=".SphereShape or .RectangleShape or None",
        init_type=".SphereShape or .RectangleShape or dict, optional",
        default="None",
    )

    @property
    def _shape_id(self) -> str:
        """
        Mitsuba shape object identifier.
        """
        return f"{self.id}_shape"

    @property
    def _bsdf_id(self) -> str:
        """
        Mitsuba BSDF object identifier.
        """
        return f"{self.id}_bsdf"

    @property
    def _template_shapes(self) -> dict:
        return {}

    @property
    def _params_shapes(self) -> dict:
        return {}

    @property
    def _template_bsdfs(self) -> dict:
        return {}

    @property
    def _params_bsdfs(self) -> dict:
        return {}

    @property
    def objects(self) -> dict[str, SceneElement]:
        # Inherit docstring
        return {
            self._shape_id: self.shape,
            f"{self._shape_id}_background": self.shape_background,
        }

    @classmethod
    def from_mesh(
        cls,
        mesh: BufferMeshShape | FileMeshShape,
        lat: pint.Quantity,
        lon: pint.Quantity,
        id: str = "surface",
        geometry: SceneGeometry = None,
        planet_radius: pint.Quantity = None,
        bsdf: BSDF = None,
    ) -> DEMSurface:
        """
        Construct a DEMSurface from a mesh object and coordinates which specify
        its location.

        Parameters
        ----------
        mesh : .BufferMeshShape or .FileMeshShape
            DEM as a triangulated mesh.

        lat : pint.Quantity
            Limits of the latitude range covered by the DEM.

        lon : pint.Quantity
            Limits of the longitude range covered by the DEM.

        id : str, optional, default: "surface"
            Identifier of the scene element.

        geometry : .SceneGeometry, optional, default: :class:`PlaneParallelGeometry() <.PlaneParallelGeometry>`
            Atmospheric geometry of the scene.

        planet_radius : pint.Quantity, optional, default: .EARTH_RADIUS
            Planet radius. Used only in case of a plane parallel geometry to
            convert between latitude/longitude and x/y coordinates.

        bsdf : .BSDF, optional, default: :class:`LambertianBSDF() <.LambertianBSDF>`
            Scattering model attached to the surface.

        Returns
        -------
        .DEMSurface
        """
        geometry = PlaneParallelGeometry() if geometry is None else geometry
        bsdf = LambertianBSDF() if bsdf is None else bsdf
        bsdf_id = f"{id}_bsdf"
        mesh = attrs.evolve(mesh, bsdf=bsdf)

        if isinstance(geometry, SphericalShellGeometry):
            if planet_radius is not None:
                warnings.warn(
                    "SphericalShellGeometry overrides the `planet_radius` argument."
                )

            opacity_mask_trafo = _to_uv(*lat.m_as(ureg.deg), *lon.m_as(ureg.deg))
            opacity_array = np.ones((3, 3))
            opacity_array[1, 1] = 0
            opacity_bitmap = mi.Bitmap(opacity_array)

            opacity_bsdf = OpacityMaskBSDF(
                id=f"{bsdf_id}_shape_background",
                nested_bsdf=bsdf,
                opacity_bitmap=opacity_bitmap,
                uv_trafo=opacity_mask_trafo,
            )

            # The 'hole' in the background surface is created at the equator:
            # we rotate it to
            trafo = (
                mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=90)
                @ mi.ScalarTransform4f.rotate(
                    axis=[0, 1, 0], angle=(90 - np.mean(lat.m_as(ureg.deg)))
                )
                @ mi.ScalarTransform4f.rotate(
                    axis=[0, 0, 1], angle=-np.mean(lon.m_as(ureg.deg))
                )
            )

            surface_background = SphereShape(
                id=f"{id}_shape_background",
                center=[0.0, 0.0, 0.0] * geometry.planet_radius.units,
                radius=geometry.planet_radius + geometry.ground_altitude,
                bsdf=opacity_bsdf,
                to_world=trafo,
            )

        elif isinstance(geometry, PlaneParallelGeometry):
            planet_radius = EARTH_RADIUS if planet_radius is None else planet_radius

            lat_length = (lat[1] - lat[0]).m_as(ureg.rad) * planet_radius
            lat_scale = (geometry.width / (lat_length * 3)).magnitude
            lon_length = (lon[1] - lon[0]).m_as(ureg.rad) * planet_radius
            lon_scale = (geometry.width / (lon_length * 3)).magnitude
            opacity_mask_trafo = mi.ScalarTransform4f.scale(
                (lon_scale, lat_scale, 1)
            ) @ mi.ScalarTransform4f.translate(
                (-0.5 + (0.5 / lon_scale), -0.5 + (0.5 / lat_scale), 0)
            )
            opacity_array = np.ones((3, 3))
            opacity_array[1, 1] = 0
            opacity_bsdf = OpacityMaskBSDF(
                nested_bsdf=bsdf,
                opacity_bitmap=opacity_array,
                uv_trafo=opacity_mask_trafo,
            )
            surface_background = RectangleShape(
                id=f"{id}_background",
                edges=geometry.width,
                center=[0.0, 0.0, geometry.ground_altitude.m]
                * geometry.ground_altitude.units,
                normal=np.array([0, 0, 1]),
                up=np.array([0, 1, 0]),
                bsdf=opacity_bsdf,
            )

        return cls(shape=mesh, shape_background=surface_background, id=id)
