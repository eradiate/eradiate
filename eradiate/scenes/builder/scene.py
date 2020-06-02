import attr
from attr.validators import deep_iterable, instance_of, optional

from .base import Instantiable, Object
from .bsdfs import BSDF
from .emitters import Emitter
from .integrators import Integrator
from .media import Medium
from .phase import PhaseFunction
from .sensors import Sensor
from .shapes import Shape
from .util import count


@attr.s
class Scene(Object, Instantiable):
    r"""This class wraps Mitsuba's ``Scene`` class.

    Class attributes:
        ``_tag`` = ``"scene"``

    Constructor arguments / public attributes:
        ``bsdfs`` (:class:`list` [:class:`eradiate.scenes.builder.BSDF`]):
            List of BSDF plugin interface objects for top-level referencing.
        ``phase`` (:class:`list` [:class:`eradiate.scenes.builder.Phase`]):
            List of phase function plugin interface objects for top-level
            referencing.
        ``media`` (:class:`list` [:class:`eradiate.scenes.builder.Medium`]):
            List of medium plugin interface objects for top-level referencing.
        ``shapes`` (:class:`list` [:class:`eradiate.scenes.builder.Shape`]):
            List of geometric shapes.
        ``emitter`` (:class:`eradiate.scenes.builder.Emitter`):
            List of phase function plugin interface objects for top-level
            referencing.
    """

    _tag = "scene"
    bsdfs = attr.ib(
        kw_only=True,
        default=[],
        converter=list,
        validator=deep_iterable(member_validator=instance_of(BSDF)),
    )
    phase = attr.ib(
        kw_only=True,
        default=[],
        converter=list,
        validator=deep_iterable(member_validator=instance_of(PhaseFunction)),
    )
    media = attr.ib(
        kw_only=True,
        default=[],
        converter=list,
        validator=deep_iterable(member_validator=instance_of(Medium)),
    )
    shapes = attr.ib(
        kw_only=True,
        default=[],
        converter=list,
        validator=deep_iterable(member_validator=instance_of(Shape)),
    )
    emitter = attr.ib(
        kw_only=True,
        default=None,
        validator=optional(instance_of(Emitter)),
    )
    sensor = attr.ib(
        kw_only=True,
        default=None,
        validator=optional(instance_of(Sensor)),
    )
    integrator = attr.ib(
        kw_only=True,
        default=None,
        validator=optional(instance_of(Integrator)),
    )

    @property
    def sequence(self):
        """Access the complete plugin interface sequence."""

        s = []
        s.extend(self.bsdfs)
        s.extend(self.phase)
        s.extend(self.media)
        s.extend(self.shapes)
        if self.emitter is not None:
            s.append(self.emitter)
        if self.sensor is not None:
            s.append(self.sensor)
        if self.integrator is not None:
            s.append(self.integrator)
        return s

    def check(self):
        """Check the scene sequence for inconsistencies.
        """
        if any([not isinstance(s, Shape) for s in self.shapes]):
            raise TypeError("all objects in 'shapes' must be Shape")
        if not count(Shape, self.shapes) and self.emitter is None:
            raise ValueError(
                "the scene must have at least one Shape or Emitter"
            )

    def to_etree(self):
        e = super().to_etree()
        self.check()
        for x in self.sequence:
            e.append(x.to_etree())
        return e

    def to_xml(self, pretty_print=False, add_version=True):
        """This method overrides ``Object``'s ``to_xml()`` with more convenient
        default values for keyword arguments.
        """
        return super().to_xml(pretty_print=pretty_print,
                              add_version=add_version)
