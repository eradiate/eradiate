import pytest

from eradiate._factory import BaseFactory

def test_register():

    class TestFactoryObject(object):
        pass

    class TestFactory(BaseFactory):
        _constructed_type = TestFactoryObject
        registry = {}

    # test successful registration
    @TestFactory.register("successful")
    class SuccessfulObject(TestFactoryObject):
        def from_dict(self):
            pass

    assert "successful" in TestFactory.registry.keys()

    # test wrong object type
    with pytest.raises(TypeError):
        @TestFactory.register("wrong_type")
        class WrongTypeObject(object):
            def from_dict(self):
                pass

    # test object without 'from_dict'
        with pytest.raises(AttributeError):
            @TestFactory.register("notfromdict")
            class NoFromDictObject(TestFactoryObject):
                pass


def test_create_convert():

    class TestFactoryObject(object):
        pass

    class TestFactory(BaseFactory):
        _constructed_type = TestFactoryObject
        registry = {}

    @TestFactory.register("subclass")
    class SubclassObject(TestFactoryObject):
        # this from_dict method is a mock up that makes it
        # easy to tell if it has been called by the factory.
        def from_dict(self):
            return "teststring"

    # assert Factory.create calls the underlying from_dict method
    assert TestFactory.create({"type": "subclass"}) == "teststring"

    # unregistered classes can't be instantiated
    with pytest.raises(ValueError):
        TestFactory.create({"type": "another_class"})

    # convert should return the object if it is not a dict
    assert TestFactory.convert("string") == "string"

    # convert should call Factory.create if a dict is passed
    assert TestFactory.convert({"type": "subclass"}) == "teststring"