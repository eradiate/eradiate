class DataGetter:
    _PATHS = None

    @classmethod
    def registered(cls):
        return list(cls._PATHS.keys())

    @classmethod
    def path(cls, id):
        try:
            return cls._PATHS[id]
        except KeyError:
            raise ValueError(f"unknown data set '{id}'")
