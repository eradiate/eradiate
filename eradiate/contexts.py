import attr
import pinttr

import eradiate
from ._attrs import documented, parse_docs
from ._units import unit_context_config as ucc
from ._units import unit_registry as ureg
from .exceptions import ModeError


@attr.s
class SpectralContext:
    """
    Context data structure holding state relevant to the evaluation of spectrally
    dependent objects.

    This object is usually used as part of a :class:`.KernelDictContext` to pass
    around spectral information to kernel dictionary emission methods which
    require spectral configuration information.

    :class:`Measures <.Measure>` also store a user-configured
    :class:`.SpectralContext` instance which drives contextual spectral
    configuration in solver applications.

    While this class is abstract, it should however be the main entry point
    to create :class:`.SpectralContext` child class objects through the
    :meth:`.SpectralContext.new` class method constructor.
    """

    @staticmethod
    def new(**kwargs):
        """
        Create a new instance of one of the :class:`SpectralContext` child
        classes. *The instantiated class is defined based on the currently active
        mode.* Keyword arguments are passed to the instantiated class's
        constructor:

        .. rubric:: Monochromatic modes [:class:`MonoSpectralContext`]

        Parameter ``wavelength`` (float):
            Wavelength. Default: 550 nm.

            Unit-enabled field (default: ucc[wavelength]).

        .. seealso::

           * :func:`eradiate.mode`
           * :func:`eradiate.set_mode`
        """
        mode = eradiate.mode()

        if mode.is_monochromatic():
            return MonoSpectralContext(**kwargs)

        raise ModeError(f"unsupported mode '{mode.id}'")

    @staticmethod
    def from_dict(d):
        """
        Create from a dictionary. This class method will additionally pre-process
        the passed dictionary to merge any field with an associated ``"_units"``
        field into a :class:`pint.Quantity` container.

        Parameter ``d`` (dict):
            Configuration dictionary used for initialisation.

        Returns → instance of cls:
            Created object.
        """

        # Pre-process dict: apply units to unit-enabled fields
        d_copy = pinttr.interpret_units(d, ureg=ureg)

        # Perform object creation
        return SpectralContext.new(**d_copy)

    @staticmethod
    def convert(value):
        """
        Object converter method.

        If ``value`` is a dictionary, this method uses :meth:`from_dict` to
        create a :class:`.SpectralContext`.

        Otherwise, it returns ``value``.
        """
        if isinstance(value, dict):
            return SpectralContext.from_dict(value)

        return value


@attr.s
class MonoSpectralContext(SpectralContext):
    wavelength = pinttr.ib(
        default=ureg.Quantity(550.0, ureg.nm),
        units=ucc.deferred("wavelength"),
    )


@parse_docs
@attr.s
class KernelDictContext:
    """
    Kernel dictionary evaluation context data structure. This class is used
    *e.g.* to store information about the spectral configuration to apply
    when generating kernel dictionaries associated with a :class:`.SceneElement`
    instance.
    """
    spectral_ctx = documented(
        attr.ib(factory=SpectralContext.new, converter=SpectralContext.convert),
        doc="Spectral context (used to evaluate quantities with any degree "
        "or kind of dependency vs spectrally varying quantities).",
        type=":class:`.SpectralContext`",
        default=":meth:`SpectralContext.new() <.SpectralContext.new>`",
    )

    ref = documented(
        attr.ib(default=True, converter=bool),
        doc="If ``True``, use references when relevant during kernel dictionary "
        "generation.",
        type="bool",
        default="True",
    )

    atmosphere_kernel_width = documented(
        pinttr.ib(default=None, units=ucc.deferred("length")),
        doc="If relevant, stores the width of the kernel object associated with "
        "the atmosphere.",
        type="float or None",
        default="None",
    )
