from eradiate.util.metaclasses import Singleton


def test_singleton():
    class MySingleton(metaclass=Singleton):
        pass

    my_singleton1 = MySingleton()
    my_singleton2 = MySingleton()
    assert my_singleton1 is my_singleton2
