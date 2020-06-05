class Singleton(type):
    """A simple singleton implementation.
    See also
    https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python.

    .. admonition:: Example

        .. code:: python

            class MySingleton(metaclass=Singleton):
                pass

            my_singleton1 = MySingleton()
            my_singleton2 = MySingleton()
            assert my_singleton1 is my_singleton2  # Should not fail
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
