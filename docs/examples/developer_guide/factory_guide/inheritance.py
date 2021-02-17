import attr

from eradiate.factory import BaseFactory


@attr.s
class MyClassBase:
    my_class_base_attr = attr.ib(default=None)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class MyClassFactory(BaseFactory):
    _constructed_type = MyClassBase
    registry = {}


@MyClassFactory.register("my_class_child")
@attr.s
class MyClassChild(MyClassBase):
    my_class_child_attr = attr.ib(default=None)


@MyClassFactory.register("my_class_child_child")
@attr.s
class MyClassChildChild(MyClassChild):
    my_class_child_child_attr = attr.ib(default=None)


@attr.s
class MyClassChildChildUnregistered(MyClassChild):
    my_class_child_child_attr = attr.ib(default=None)


print(MyClassChild())
# MyClassChild(my_class_base_attr=None, my_class_child_attr=None)
print(MyClassChildChild())
# MyClassChildChild(my_class_base_attr=None, my_class_child_attr=None, my_class_child_child_attr=None)
print(MyClassFactory.registry)
# {'my_class_child': <class '__main__.MyClassChild'>, 'my_class_child_child': <class '__main__.MyClassChildChild'>}
