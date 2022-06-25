from ...util import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submod_attrs={
        "_core": [
            "Canopy",
            "CanopyElement",
            "InstancedCanopyElement",
            "biosphere_factory",
        ],
        "_discrete": ["DiscreteCanopy"],
        "_leaf_cloud": ["LeafCloud"],
        "_tree": ["AbstractTree", "MeshTree", "MeshTreeElement"],
    },
)

del lazy_loader
