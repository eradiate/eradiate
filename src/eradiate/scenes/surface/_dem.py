from __future__ import annotations

import typing as t
import warnings

import attrs
import mitsuba as mi
import numpy as np
import pint
import pinttrs
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
from ...attrs import define, documented, get_doc
from ...constants import EARTH_RADIUS
from ...units import symbol, to_quantity
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


def _middle(a, axis=None):
    a_min = a.min(axis=axis)
    a_max = a.max(axis=axis)
    return 0.5 * (a_min + a_max)


def _apply(transform: "mitsuba.ScalarTransform4f", vertices):
    """
    Apply a Mitsuba transform to a Numpy buffer. This function works also with
    scalar variants of Mitsuba.
    """
    return (
        transform.matrix.numpy()
        @ np.concatenate((vertices, np.ones((vertices.shape[0], 1))), axis=1).T
    ).T[:, :3]


def _mercator(lon, lat, planet_radius):
    """
    Convert longitude and latitude values to (x, y) coordinates using a
    Mercator projection.
    """
    # See math here: https://en.wikipedia.org/wiki/Mercator_projection
    x = planet_radius * lon
    y = planet_radius * np.log(np.tan(0.25 * np.pi + 0.5 * lat))
    return x, y


def _mercator_inverse(x, y, planet_radius):
    """
    Convert (x, y) coordinates to longitude and latitude using an inverse
    Mercator projection.
    """
    # See math here: https://en.wikipedia.org/wiki/Mercator_projection
    lon = x / planet_radius
    lat = 2.0 * np.arctan(np.exp(y / planet_radius)) - 0.5 * np.pi
    return lon, lat


def _transform_vertices_spherical_shell_lonlat(vertices, planet_radius):
    """
    Convert the (lon, lat, elevation) vertices from the initial vertex generation
    into (x, y, z) values for spherical shell geometries.

    Parameters
    ----------
    vertices : ndarray
        List of mesh vertices in (lon, lat, elevation) tuples, with lon and lat
        in radians.

    planet_radius : float
        Planet radius in kernel length units.

    Returns
    -------
    vertices : ndarray
    """
    lon = vertices[:, 0]
    lat = vertices[:, 1]
    lon_center, lat_center = _middle(vertices[:, :2], axis=0)
    elevation = vertices[:, 2]

    phi_r = lon
    theta_r = np.pi / 2.0 - lat

    x = np.sin(theta_r) * np.cos(phi_r) * (elevation + planet_radius)
    y = np.sin(theta_r) * np.sin(phi_r) * (elevation + planet_radius)
    z = np.cos(theta_r) * (elevation + planet_radius)
    vertices = np.array((x, y, z)).transpose()

    # At this point, vertices are positioned in an ECEF frame; transform them to
    # the local ENU frame
    trafo = _transform_lonlat_range_to_local(lon_center, lat_center)
    vertices = _apply(trafo, vertices)
    return vertices


def _transform_lonlat_range_to_local(
    lon_center, lat_center
) -> "mitsuba.ScalarTransform4f":
    """
    Create a transformation matrix to position points placed in ECEF coordinates
    from longitude / latitude information to the local frame in a spherical-shell
    geometry.
    """
    angle_1 = np.rad2deg(lon_center)
    angle_2 = 90.0 - np.rad2deg(lat_center)
    angle_3 = 90.0
    return (
        mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=-angle_3)
        @ mi.ScalarTransform4f.rotate(axis=[0, 1, 0], angle=-angle_2)
        @ mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=-angle_1)
    )


def triangulate_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None = None,
    flip: bool = False,
    divide: t.Literal["nesw", "nwse"] = "nesw",
):
    """
    Create a 2D triangulation for a regular (x, y) grid.

    Parameters
    ----------
    x : ndarray
        List of x grid values.

    y : ndarray
        List of y grid values.

    z : ndarray, optional
        If passed, a 3rd coordinate for vertices in gridded format (y-major).

    flip : bool
        If ``True``, flip triangle orientation (clockwise by default).

    divide : {"nesw", "nwse"}, default: "nesw"
        Cell division method.

    Returns
    -------
    vertices : ndarray
        Vertex list (y-major) as a (n, 2) array.

    faces : ndarray
        Face definitions as a (n, 3) array.
    """
    x_grid, y_grid = np.meshgrid(x, y)
    vertices = np.array((x_grid.ravel(), y_grid.ravel())).T

    xi = np.arange(0, len(x), 1, dtype="int")
    yi = np.arange(0, len(y), 1, dtype="int")

    def _vertex_indices(xg, yg, xstride):
        return xg + xstride * yg

    xg, yg = np.meshgrid(xi[:-1], yi[:-1])
    vertices_sw = _vertex_indices(xg.ravel(), yg.ravel(), xstride=len(x))

    xg, yg = np.meshgrid(xi[1:], yi[:-1])
    vertices_se = _vertex_indices(xg.ravel(), yg.ravel(), xstride=len(x))

    xg, yg = np.meshgrid(xi[:-1], yi[1:])
    vertices_nw = _vertex_indices(xg.ravel(), yg.ravel(), xstride=len(x))

    xg, yg = np.meshgrid(xi[1:], yi[1:])
    vertices_ne = _vertex_indices(xg.ravel(), yg.ravel(), xstride=len(x))

    if divide == "nesw":
        face_indices_1 = np.array((vertices_sw, vertices_se, vertices_ne)).T
        face_indices_2 = np.array((vertices_sw, vertices_ne, vertices_nw)).T
    elif divide == "nwse":
        face_indices_1 = np.array((vertices_sw, vertices_nw, vertices_se)).T
        face_indices_2 = np.array((vertices_nw, vertices_ne, vertices_se)).T
    else:
        raise ValueError(f"unknown cell division method '{divide}'")

    faces = np.concatenate((face_indices_1, face_indices_2))

    if flip:
        faces = faces[:, [0, 2, 1]]

    # If relevant, add elevation as 3rd vertex coordinate
    if z is not None:
        # IMPORTANT: meshgrid() produces an (N_y, N_x)-shaped array, while we
        # expected the z array to be laid out oppositely; hence the transpose
        # prior to flattening
        vertices = np.concatenate((vertices, z.T.ravel().reshape(-1, 1)), axis=1)

    return vertices, faces


def _dem_texcoords(xlon, ylat, xlon_lim, ylat_lim) -> np.ndarray:
    """
    Map the x and y coordinates of the elements of `vertices` to the [0, 1] range
    using a linear scale within xlim and ylim respectively.

    Parameters
    ----------
    xlon : array-like
        An array of shape (N,) that contains the x/longitude coordinates of an
        arbitrary number of vertices.

    ylat : array-like
        An array of shape (N,) that contains the y/latitude coordinates of an
        arbitrary number of vertices.

    xlon_lim : tuple[float, float]
        Lower and upper bounds defining the linear transform that converts the x
        coordinate to the u texture coordinate.

    ylat_lim : tuple[float, float]
        Lower and upper bounds defining the linear transform that converts the y
        coordinate to the v texture coordinate.
    """
    uvs_x = (xlon - xlon_lim[0]) / (xlon_lim[1] - xlon_lim[0])
    uvs_y = (ylat - ylat_lim[0]) / (ylat_lim[1] - ylat_lim[0])

    return np.stack([uvs_x, uvs_y]).T


def mesh_from_dem(
    da: xr.DataArray,
    geometry: str | dict | SceneGeometry,
    planet_radius: pint.Quantity | float | None = None,
    add_texcoords: bool = False,
) -> tuple[BufferMeshShape, pint.Quantity, pint.Quantity]:
    """
    Construct a DEM surface mesh from a data array holding elevation data.

    This function has 4 modes of operations, depending on 2 parameters:

    * the selected geometry type (plane-parallel or spherical-shell);
    * the coordinates of the input dataset (longitude/latitude or x/y).

    Alongside the generated mesh, this function returns extents that can be used
    to position a background stencil around the produced shape (see the notes
    for details).

    Parameters
    ----------
    da : DataArray
        Data array with elevation data, indexed either by latitude and longitude
        coordinates or x and y coordinates.

    geometry : .SceneGeometry or dict or str
        Scene geometry configuration. The value is pre-processed by the
        :meth:`.SceneGeometry.convert` converter.

    planet_radius : quantity or float, default: :data:`.EARTH_RADIUS`
        Planet radius used to convert latitude/longitude to x/y when
        ``geometry`` is a :class:`.PlaneParallelGeometry` instance.
        This parameter is unused otherwise. If a unitless value is passed, it is
        interpreted using
        :ref:`default config length units <sec-user_guide-unit_guide_user>`.

    add_texcoords : bool, default: False
        If ``True``, texture coordinates are added to the created mesh.

    Returns
    -------
    mesh : .BufferMeshShape
        A triangulated mesh representing the DEM.

    xlon_lim : quantity
        Limits of DEM on the x/longitude axis, in length (resp. angle) units for
        plane-parallel (resp. spherical-shell) geometries.

    ylat_lim : quantity
        Limits of DEM on the y/latitude axis, in length (resp. angle) units for
        plane-parallel (resp. spherical-shell) geometries.

    Notes
    -----
    * The ``da`` parameter may use the following formats:

      * with longitude / latitude coordinates, then named ``"lon"`` and
        ``"lat"`` respectively;
      * with x / y coordinates, then named ``"x"`` and ``"y"`` respectively.

      Coordinate and variable units are specified using the ``units`` xarray
      attributes and are expected to be consistent with coordinate names.

    * The mesh generation algorithm operates in four modes, two of which exist
      for testing purposes and not recommended for quantitative applications:

      * **Plane-parallel / xy mode:** The mesh is generated using vertices
        positioned at grid points, then offset to center it at the origin
        location of the local frame. The returned extent is in length units.
      * **Spherical-shell / lonlat mode:** Coordinates are straightforwardly
        converted to an ECEF frame. The resulting mesh is positioned where it
        should fit on the reference geoid (currently a perfect sphere). The
        returned extent is in longitude/latitude.
      * **Plane-parallel / lonlat mode:** The input data is interpreted using a
        Mercator projection, and generated vertices are offset to center the
        mesh at the origin of the local frame. The returned extent is in length
        units. *The projection is highly distorting and should not be used for
        quantitative applications.*
      * **Spherical-shell / xy mode:** Coordinates are converted to longitude
        and latitude using an inverse Mercator projection, assuming that the
        dataset is centred at the equator, then reinterpreted in an ECEF frame.
        The returned extent is in longitude/latitude. *The projection is highly
        distorting and should not be used for quantitative applications.*

    * The generated mesh can optionally be assigned texture coordinates used to
      map spatially-varying data (*e.g.* textured reflectance value for a
      Lambertian BSDF).
    """
    # Pre-process geometry parameter
    geometry = SceneGeometry.convert(geometry)

    if planet_radius is not None:
        if isinstance(geometry, SphericalShellGeometry):
            warnings.warn(
                "The ``planet_radius`` argument is set to a user-defined "
                "value but will be overridden by the value held by the "
                "SphericalGeometry instance passed as the ``geometry`` "
                "parameter."
            )

        # Add default units if quantity is unitless
        planet_radius = pinttrs.util.ensure_units(planet_radius, ucc.get("length"))

    # Set default planet radius value
    planet_radius = EARTH_RADIUS if planet_radius is None else planet_radius
    if isinstance(geometry, SphericalShellGeometry):
        planet_radius = geometry.planet_radius

    # Check data array coordinates and dimensions
    length_kernel_u = uck.get("length")

    if ("lon" in da.coords) and ("lat" in da.coords):
        mode = "lonlat"
        xlon_dim, ylat_dim = "lon", "lat"

    elif ("x" in da.coords) and ("y" in da.coords):
        mode = "xy"
        xlon_dim, ylat_dim = "x", "y"

    else:
        raise ValueError(
            "Data array coordinates must include either `x/y` or `lon/lat`.\n"
            f"Got: {da.coords}"
        )

    # Convert to distances in plane-parallel geometry,
    # and to angles in spherical-shell geometry.
    xlon = to_quantity(da[xlon_dim])
    ylat = to_quantity(da[ylat_dim])
    xlon_center = _middle(xlon)
    ylat_center = _middle(ylat)
    # Extract elevation data and ensure y-major layout
    elevation = to_quantity(da.transpose(xlon_dim, ylat_dim))
    # By default, no texture coordinates are assigned
    texcoords = None

    # Process vertex data depending on geometry and coordinate mode, generate triangulation
    if isinstance(geometry, PlaneParallelGeometry):
        if mode == "xy":
            xlon = xlon - xlon_center
            ylat = ylat - ylat_center

            vertices, faces = triangulate_grid(
                xlon.m_as(length_kernel_u),
                ylat.m_as(length_kernel_u),
                elevation.m_as(length_kernel_u),
            )

            xlon_lim = (xlon.m.min(), xlon.m.max()) * xlon.u
            ylat_lim = (ylat.m.min(), ylat.m.max()) * ylat.u

            # If relevant, assign texture coordinates
            if add_texcoords:
                texcoords = _dem_texcoords(
                    vertices[:, 0], vertices[:, 1], xlon_lim.m, ylat_lim.m
                )

        elif mode == "lonlat":
            x, y = _mercator(
                xlon.m_as(ureg.rad),
                ylat.m_as(ureg.rad),
                planet_radius.m_as(length_kernel_u),
            )
            da = (
                da.assign_coords(
                    {
                        "x": ("lon", x, {"units": symbol(length_kernel_u)}),
                        "y": ("lat", y, {"units": symbol(length_kernel_u)}),
                    }
                )
                .swap_dims({"lon": "x", "lat": "y"})
                .drop_vars(("lon", "lat"))
            )
            return mesh_from_dem(
                da, geometry, planet_radius, add_texcoords=add_texcoords
            )

        else:
            raise RuntimeError(f"unknown input mode {mode}")

    elif isinstance(geometry, SphericalShellGeometry):
        if mode == "xy":
            lon, lat = _mercator_inverse(
                xlon.m_as(length_kernel_u),
                ylat.m_as(length_kernel_u),
                planet_radius.m_as(length_kernel_u),
            )
            da = (
                da.assign_coords(
                    {
                        "lon": ("x", np.rad2deg(lon), {"units": "degree"}),
                        "lat": ("y", np.rad2deg(lat), {"units": "degree"}),
                    }
                )
                .swap_dims(({"x": "lon", "y": "lat"}))
                .drop_vars(("x", "y"))
            )
            return mesh_from_dem(
                da, geometry, planet_radius, add_texcoords=add_texcoords
            )

        elif mode == "lonlat":
            xlon = xlon.to(ureg.rad)
            ylat = ylat.to(ureg.rad)
            vertices, faces = triangulate_grid(
                xlon.m, ylat.m, elevation.m_as(length_kernel_u)
            )

            # If relevant, assign texture coordinates
            xlon_lim = (xlon.m.min(), xlon.m.max()) * xlon.u
            ylat_lim = (ylat.m.min(), ylat.m.max()) * ylat.u
            if add_texcoords:
                texcoords = _dem_texcoords(
                    vertices[:, 0], vertices[:, 1], xlon_lim.m, ylat_lim.m
                )

            # Rotate mesh to Eradiate's local frame (located at the North pole)
            vertices = _transform_vertices_spherical_shell_lonlat(
                vertices, planet_radius.m_as(length_kernel_u)
            )
            # Recompute limits (returned afterwards)
            xlon_lim = (xlon.m.min(), xlon.m.max()) * xlon.u
            ylat_lim = (ylat.m.min(), ylat.m.max()) * ylat.u

        else:
            raise RuntimeError(f"unknown input mode {mode}")

    else:  # For completeness
        raise TypeError(f"unhandled geometry type '{type(PlaneParallelGeometry)}'")

    # Create mesh instance
    mesh = BufferMeshShape(vertices=vertices, faces=faces, texcoords=texcoords)

    return mesh, xlon_lim, ylat_lim


@define(eq=False, slots=False)
class DEMSurface(Surface):
    """
    DEM Surface [``dem``]

    A mesh-based representation of a Digital Elevation Model (DEM). This class
    holds a mesh shape instance, as well as a background shape that provides a
    surface beyond the extents of the mesh.

    The intended instantiation method is to, first, create a mesh using the
    :func:`.mesh_from_dem`, then use the data it returns to call the
    :meth:`.from_mesh` constructor.
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
        init_type=".BufferMeshShape or .FileMeshShape or dict",
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
        xlon_lim: pint.Quantity,
        ylat_lim: pint.Quantity,
        id: str = "surface",
        geometry: SceneGeometry | str | dict = "plane_parallel",
        planet_radius: pint.Quantity | None = None,
        bsdf: BSDF | None = None,
        bsdf_mesh: BSDF | None = None,
        bsdf_background: BSDF | None = None,
    ) -> DEMSurface:
        """
        Construct a :class:`.DEMSurface` instance from a mesh object and
        coordinates which specify its location.

        Parameters
        ----------
        mesh : .BufferMeshShape or .FileMeshShape
            DEM as a triangulated mesh. The BSDF of this shape will be
            overridden by the ``bsdf_mesh`` parameter.

        xlon_lim : quantity
            Limits of the x/longitude range covered by the mesh.

        ylat_lim : quantity
            Limits of the y/latitude range covered by the mesh.

        id : str, default: "surface"
            Identifier of the scene element.

        geometry : .SceneGeometry or str or dict, default: "plane_parallel"
            Scene geometry. Strings and dictionaries are processed by the
            :meth:`.SceneGeometry.convert` function.

        planet_radius : quantity, default: .EARTH_RADIUS
            Planet radius. Used only in case of a plane parallel geometry to
            convert between latitude/longitude and x/y coordinates.

        bsdf : .BSDF, default: :class:`LambertianBSDF() <.LambertianBSDF>`
            Alias to ``bsdf_mesh`` (**deprecated**). If both are defined,
            ``bsdf_mesh`` takes precedence.

        bsdf_mesh : .BSDF, default: :class:`LambertianBSDF() <.LambertianBSDF>`
            Scattering model attached to the mesh.

        bsdf_background : .BSDF, default: :class:`LambertianBSDF() <.LambertianBSDF>`
            Scattering model attached to the background shape.

        Returns
        -------
        .DEMSurface
        """
        geometry = SceneGeometry.convert(geometry)

        if bsdf_mesh is None:
            bsdf_mesh = LambertianBSDF() if bsdf is None else bsdf

        else:
            if bsdf is not None:
                warnings.warn(
                    "while calling DEMSurface.from_mesh(): "
                    "both bsdf and bsdf_mesh parameters were specified; "
                    "bsdf_mesh takes precedence"
                )

        bsdf_background = (
            LambertianBSDF() if bsdf_background is None else bsdf_background
        )

        mesh = attrs.evolve(mesh, bsdf=bsdf_mesh)

        if isinstance(geometry, PlaneParallelGeometry):
            kernel_length_units = uck.get("length")
            g_width = geometry.width.to(kernel_length_units)
            g_altitude = geometry.ground_altitude.to(kernel_length_units)

            x_range = (xlon_lim[1] - xlon_lim[0]).m_as(kernel_length_units)
            x_scale = g_width.m / (x_range * 3.0)

            y_range = (ylat_lim[1] - ylat_lim[0]).m_as(kernel_length_units)
            y_scale = g_width.m / (y_range * 3.0)

            opacity_array = np.array(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
            )
            opacity_mask_trafo = mi.ScalarTransform4f.scale(
                [x_scale, y_scale, 1.0]
            ) @ mi.ScalarTransform4f.translate(
                [-0.5 + (0.5 / x_scale), -0.5 + (0.5 / y_scale), 0.0]
            )
            opacity_bsdf = OpacityMaskBSDF(
                nested_bsdf=bsdf_background,
                opacity_bitmap=opacity_array,
                uv_trafo=opacity_mask_trafo,
            )
            surface_background = RectangleShape(
                id=f"{id}_background",
                edges=g_width,
                center=[0.0, 0.0, g_altitude.m] * g_altitude.units,
                normal=[0, 0, 1],
                up=[0, 1, 0],
                bsdf=opacity_bsdf,
            )

        elif isinstance(geometry, SphericalShellGeometry):
            if planet_radius is not None:
                warnings.warn(
                    "SphericalShellGeometry overrides the `planet_radius` argument."
                )

            def _to_uv(lon_lim, lat_lim) -> "mitsuba.ScalarTransform4f":
                """
                Compute the `to_uv` transformation for the opacity mask bitmap.
                It moves the central (transparent) part of the bitmap to where
                it covers the specified longitude/latitude extent.
                """
                lon_range = lon_lim[1] - lon_lim[0]
                lon_scale = 120.0 / lon_range

                lat_range = lat_lim[1] - lat_lim[0]
                lat_scale = 60.0 / lat_range

                lon_middle = _middle(lon_lim)
                lon_uv = lon_middle / 360.0 + 0.5

                lat_mean = (lat_lim[1] + lat_lim[0]) / 2.0
                lat_uv = 0.5 - (lat_mean / 180)

                return mi.ScalarTransform4f.scale(
                    (lon_scale, lat_scale, 1.0)
                ) @ mi.ScalarTransform4f.translate(
                    [-lon_uv + (0.5 / lon_scale), -lat_uv + (0.5 / lat_scale), 0.0]
                )

            opacity_mask_trafo = _to_uv(
                xlon_lim.m_as(ureg.deg), ylat_lim.m_as(ureg.deg)
            )
            opacity_array = np.array(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
            )
            opacity_bsdf = OpacityMaskBSDF(
                nested_bsdf=bsdf_background,
                opacity_bitmap=opacity_array,
                uv_trafo=opacity_mask_trafo,
            )

            # The 'hole' in the background surface is created at the equator:
            # rotate the shape to align it with the mesh
            trafo = (
                mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=90)
                @ mi.ScalarTransform4f.rotate(
                    axis=[0, 1, 0], angle=(90 - _middle(ylat_lim.m_as(ureg.deg)))
                )
                @ mi.ScalarTransform4f.rotate(
                    axis=[0, 0, 1], angle=-_middle(xlon_lim.m_as(ureg.deg))
                )
            )

            surface_background = SphereShape(
                id=f"{id}_shape_background",
                center=[0.0, 0.0, 0.0] * geometry.planet_radius.units,
                radius=geometry.planet_radius + geometry.ground_altitude,
                bsdf=opacity_bsdf,
                to_world=trafo,
            )

        else:
            raise TypeError(f"Unhandled scene geometry type '{type(geometry)}'")

        return cls(shape=mesh, shape_background=surface_background, id=id)
