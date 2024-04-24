Digital elevation model support
===============================

The :class:`.DEMExperiment` class extends the capabilities of the
:class:`.AtmosphereExperiment` class with digital elevation model support. This
class supports all the parameters allowed by :class:`.AtmosphereExperiment`,
except for the ``surface`` parameter, which accepts a :class:`.DEMSurface`
instance. It should be noted that any experiment performed with this class is
3D.

A :class:`.DEMSurface` features two components:

* a main terrain shape of a finite horizontal extent, consisting of a
  triangulated mesh;
* a background shape, which defines the surface outside of the terrain's extent.

Background control is limited to setting the associated reflection model. The
terrain shape and

The DEM can be defined in several ways and it can be used with any surface
reflection model. The DEM dataset is converted into a 3D shape and integrated in
a background 1D experiment. Eradiate makes sure that the 3D model derived from
the DEM dataset is integrated in the background surface in a watertight fashion
so as to avoid any energy leakage that would occur if rays could escape under the
terrain model.

The DEM geometry is defined using one of the mesh classes
(:class:`.FileMeshShape` if loaded from disk, or :class:`.BufferMeshShape` if
initialized from an in-memory buffer). The :func:`.dem_from_mesh` helper
facilitates the operation of building a triangulated mesh in memory from a NetCDF
dataset.