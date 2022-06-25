"""The Eradiate radiative transfer simulation software package."""


from ._version import _version

__version__ = _version  #: Eradiate version string.

# -- Lazy imports ------------------------------------------------------

from .util import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules=[
        "ckd",
        "contexts",
        "converters",
        "data",
        "experiments",
        "kernel",
        "notebook",
        "pipelines",
        "plot",
        "rng",
        "scenes",
        "units",
        "validators",
        "xarray",
    ],
    submod_attrs={
        "_config": ["config"],
        "_mode": [
            "mode",
            "modes",
            "set_mode",
            "supported_mode",
            "unsupported_mode",
            "Mode",
            "ModeFlags",
        ],
        "experiments": ["run"],
        "notebook": ["load_ipython_extension"],
        "units": ["unit_registry", "unit_context_config", "unit_context_kernel"],
    },
)

del lazy_loader
