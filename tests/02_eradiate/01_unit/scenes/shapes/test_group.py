import mitsuba as mi
from rich.pretty import pprint

from eradiate.contexts import KernelDictContext
from eradiate.kernel import mi_traverse
from eradiate.scenes.bsdfs import LambertianBSDF
from eradiate.scenes.core import Ref, Scene, traverse
from eradiate.scenes.shapes import GroupShape, RectangleShape
from eradiate.test_tools.types import check_scene_element


def test_group_construct(mode_mono):
    group = GroupShape(
        shapes={
            "rectangle_1": RectangleShape(bsdf=Ref(id="the_bsdf")),
            "rectangle_2": RectangleShape(bsdf=Ref(id="the_bsdf")),
        },
        bsdfs={"the_bsdf": LambertianBSDF()},
    )

    kdict_template, umap_template = traverse(Scene(objects={"the_group": group}))
    mi_scene = mi_traverse(
        mi.load_dict(kdict_template.render(ctx=KernelDictContext())),
        umap_template=umap_template,
    )

    check_scene_element(group)
