from ...util import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submod_attrs={
        "_core": ["Shape", "shape_factory"],
        "_cuboid": ["CuboidShape"],
        "_rectangle": ["RectangleShape"],
        "_sphere": ["SphereShape"],
        "_filemesh": ["FileMeshShape"],
        "_buffermesh": ["BufferMeshShape"],
    },
)

del lazy_loader
