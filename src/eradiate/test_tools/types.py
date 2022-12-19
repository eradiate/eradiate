import typing as t

import mitsuba as mi

from ..contexts import KernelDictContext
from ..scenes.core import NodeSceneElement, traverse


def check_type(
    cls,
    expected_mro: t.Optional[t.List[t.Type]] = None,
    expected_slots: t.Optional[t.Sequence[str]] = None,
) -> None:
    """
    Check if a type has the appropriate MRO and slots. Failed checks result in a
    comprehensive assert error.

    Parameters
    ----------
    cls : type
        Type to check.

    expected_mro : list[type], optional
        Expected method resolution order. The expected MRO can be a subsequence
        of the actual MRO. This argument is typically used to check if a
        superclass is resolved before another. If unset, no MRO check is done.

    expected_slots : list[str], optional
        Expected slots, if any. An empty sequence may be passed to specify that
        the type should have no slots. If unset, no slot check is done.
    """

    if expected_mro is not None:
        mro = [x for x in cls.__mro__ if x in expected_mro]
        assert mro == expected_mro, (
            f"MRO of '{cls.__name__}' should be "
            f"{[x.__name__ for x in expected_mro]}, not {[x.__name__ for x in mro]}; "
            "check inheritance order"
        )

    if expected_slots is not None:
        expected_slots = set(expected_slots)
        slots = set(getattr(cls, "__slots__", set()))
        assert slots == expected_slots, (
            f"Class '{cls.__name__}' should have "
            + (f"no slots" if not expected_slots else f"slots {expected_slots}")
            + (" but has none" if not slots else f", not {slots}")
        )


def check_node_scene_element(instance: NodeSceneElement, mi_cls: "mitsuba.Object"):
    """
    Perform kernel dictionary checks on a node scene element.

    This function checks if the node scene element can produce a valid kernel
    dictionary template, as well as an appropriate parameter table.

    The returned Mitsuba object and parameter table can be used to perform
    additional checks.

    Parameters
    ----------
    instance : :class:`.NodeSceneElement`
        Node scene element to check.

    mi_cls : :class:`mitsuba.Object`
        Mitsuba class the node scene element expands to.

    Returns
    -------
    mi_obj : :class:`mitsuba.Object`
        Mitsuba object the node scene element expands to.

    mi_params : dict
        Parameter table of the Mitsuba object.
    """
    template, params = traverse(instance)

    # Check if the template can be instantiated
    ctx = KernelDictContext()
    kernel_dict = template.render(ctx)
    mi_obj = mi.load_dict(kernel_dict)
    assert isinstance(mi_obj, mi_cls)

    # Check if parameters can be updated
    kernel_params = params.render(ctx)
    mi_params = mi.traverse(mi_obj)

    for key, value in kernel_params.items():
        mi_params[key] = value

    mi_params.update()

    return mi_obj, mi_params
