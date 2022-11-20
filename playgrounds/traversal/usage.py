import mitsuba as mi
import traversal
from rich import print

mi.set_variant("scalar_mono_double")


# Object hierarchy basics
rectangle = traversal.RectangleShape(
    bsdf=traversal.DiffuseBSDF(
        reflectance=traversal.InterpolatedSpectrum(
            [300.0, 700.0],
            [0.4, 0.6],
        )
    )
)
print(rectangle)

# Basic traversal, dict template rendering
kernel_dict = traversal.traverse(rectangle)
print(kernel_dict)
print(kernel_dict.render(w=500.0))

# Partial scene element support
two_rectangles = traversal.MultiShape(
    shapes=[
        traversal.RectangleShape(),
        traversal.RectangleShape(),
    ]
)
scene = traversal.Scene(partials=[two_rectangles])
kernel_dict = traversal.traverse(scene)
mi.load_dict(kernel_dict.render(w=500))

# Processing (parametric loop)
exp = traversal.SomeExperiment(
    shape=rectangle,
    measures=[
        traversal.PerspectiveCamera(),
        traversal.PerspectiveCamera(),
    ],
)
print(exp.process())
