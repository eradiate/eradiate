class DataGetter:
    """Base class for data getters."""

    _PATHS = None
    """Dictionary mapping registered resource IDs with paths (or globs).
    """

    @classmethod
    def registered(cls):
        """Return list of registered resources."""
        return list(cls._PATHS.keys())

    @classmethod
    def path(cls, id):
        """Get paths to registered resources from _PATHS."""
        try:
            return cls._PATHS[id]
        except KeyError:
            raise ValueError(f"unknown data set '{id}'")

    @classmethod
    def open(cls, id):
        """Open requested data set. No default implementation."""
        raise NotImplementedError
