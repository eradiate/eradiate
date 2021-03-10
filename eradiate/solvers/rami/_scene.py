import attr
import numpy as np

from eradiate import unit_context_kernel as uck, validators
from eradiate._attrs import documented, get_doc, parse_docs
from eradiate.scenes.biosphere import BiosphereFactory, Canopy
from eradiate.scenes.core import KernelDict
from eradiate.scenes.integrators import Integrator, IntegratorFactory, PathIntegrator
from eradiate.scenes.measure._distant import DistantMeasure
from eradiate.scenes.surface import LambertianSurface, Surface, SurfaceFactory
from eradiate.solvers.core._scene import Scene


@parse_docs
@attr.s
class RamiScene(Scene):
    """
    Scene abstraction suitable for radiative transfer simulation on RAMI
    benchmark scenes.
    """

    surface = documented(
        attr.ib(
            factory=LambertianSurface,
            converter=SurfaceFactory.convert,
            validator=attr.validators.instance_of(Surface),
        ),
        doc="Surface specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by "
        ":meth:`SurfaceFactory.convert() <.SurfaceFactory.convert>`. "
        ".. note:: Surface size will be overridden using canopy parameters.",
        type=":class:`.Surface` or dict",
        default=":class:`LambertianSurface() <.LambertianSurface>`",
    )

    canopy = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(BiosphereFactory.convert),
            validator=attr.validators.optional(attr.validators.instance_of(Canopy)),
        ),
        doc="Canopy specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by "
        ":meth:`BiosphereFactory.convert() <.BiosphereFactory.convert>`.",
        type=":class:`.Canopy` or dict",
        default=":class:`HomogeneousDiscreteCanopy() <.HomogeneousDiscreteCanopy>`",
    )

    padding = documented(
        attr.ib(
            default=0,
            converter=int,
            validator=validators.is_positive
        ),
        doc="Padding level. The scene will be padded with copies to account for "
        "adjacency effects. This, in practice, has effects similar to "
        "making the scene periodic."
        "A value of 0 will yield only the defined scene. A value of 1 "
        "will add one copy in every direction, yielding a 3×3 patch. A "
        "value of 2 will yield a 5×5 patch, etc. The optimal padding level "
        "depends on the scene.",
        type="int",
        default="0",
    )

    integrator = documented(
        attr.ib(
            factory=PathIntegrator,
            converter=IntegratorFactory.convert,
            validator=attr.validators.instance_of(Integrator),
        ),
        doc=get_doc(Scene, attrib="integrator", field="doc"),
        type=get_doc(Scene, attrib="integrator", field="type"),
        default=":class:`PathIntegrator() <.PathIntegrator>`",
    )

    def __attrs_post_init__(self):
        # Parts of the init sequence we couldn't take care of using converters

        # Don't forget to call super class post-init
        super(RamiScene, self).__attrs_post_init__()

        # Override surface width with canopy width
        if self.canopy is not None:
            self.surface.width = max(self.canopy.size[:2])

        # Scale surface to accomodate padding
        self.surface.width = self.surface.width * (1.0 + 2.0 * self.padding)

        # Process measures
        for measure in self.measures:
            # Override ray target location if relevant
            if isinstance(measure, DistantMeasure):
                if measure.target is None:
                    if self.canopy is not None:
                        measure.target = dict(
                            type="rectangle",
                            xmin=-0.5 * self.canopy.size[0],
                            xmax=0.5 * self.canopy.size[0],
                            ymin=-0.5 * self.canopy.size[1],
                            ymax=0.5 * self.canopy.size[1],
                        )
                    else:
                        measure.target = dict(
                            type="rectangle",
                            xmin=-0.5 * self.surface.width,
                            xmax=0.5 * self.surface.width,
                            ymin=-0.5 * self.surface.width,
                            ymax=0.5 * self.surface.width,
                        )

    def kernel_dict(self, ref=True):
        from eradiate.kernel.core import ScalarTransform4f

        result = KernelDict.empty()

        if self.canopy is not None:
            canopy_dict = self.canopy.kernel_dict(ref=True)
            result.add(canopy_dict)

            # specify the instances for padding
            patch_size = max(self.canopy.size[:2])
            kdu_length = uck.get("length")
            for shapegroup_id in canopy_dict.keys():
                if shapegroup_id.find("bsdf") != -1:
                    continue
                for x_offset in np.arange(-self.padding, self.padding + 1):
                    for y_offset in np.arange(-self.padding, self.padding + 1):
                        instance_dict = {
                            "type": "instance",
                            "group": {
                                "type": "ref",
                                "id": f"{shapegroup_id}"
                            },
                            "to_world": ScalarTransform4f.translate([
                                patch_size.m_as(kdu_length) * x_offset,
                                patch_size.m_as(kdu_length) * y_offset,
                                0.0
                            ])

                        }
                        result[f"instance{x_offset}_{y_offset}"] = instance_dict

        result.add([
            self.surface,
            self.illumination,
            *self.measures,
            self.integrator,
        ])

        return result