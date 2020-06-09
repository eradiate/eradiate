""" Eradiate scene generation library """

import eradiate.kernel
from .base import SceneHelper
from ..util.exceptions import KernelVariantError


class SceneDict(dict):
    """This class adds metadata and checks to help generate scene specification
    dictionaries appropriate for instantiation with
    :func:`mitsuba.core.xml.load_dict`."""

    def __init__(self, *args, **kwargs):
        """Initialise self and set :attr:`variant` attribute based on currently
        set variant.

        Raises → :class:`~eradiate.util.exceptions.KernelVariantError`
            If no kernel variant is set.
        """
        super().__init__(*args, **kwargs)

        variant = eradiate.kernel.variant()

        if variant is not None:
            self.variant = variant  #: Kernel variant with for which the scene is created
        else:
            raise KernelVariantError("a kernel variant must be selected to "
                                     "create a SceneDict object")

    @classmethod
    def empty(cls):
        """Create an empty scene."""
        return cls({"type": "scene"})

    def check(self):
        """Perform basic checks on the dictionary:

        - check that the ``{"type": "scene"}`` parameter is included;
        - check if the variant for which the scene dictionary was created is
          the same as the current one.

        Raises → ValueError
            If  the ``{"type": "scene"}`` parameter is missing.

        Raises → :class:`.KernelVariantError`
            If the variant for which the scene dictionary was created is
            not the same as the current one
         """
        if self.get("type", None) != "scene":
            raise ValueError("scene dictionary is missing {'type': 'scene'} "
                             "parameters")

        variant = eradiate.kernel.variant()
        if self.variant != variant:
            raise KernelVariantError(f"scene dictionary created for kernel "
                                     f"variant '{self.variant}', incompatible "
                                     f"with current variant '{variant}'")

    def add(self, content):
        """Merge the content of a :class:`~eradiate.scenes.base.SceneHelper` or
        another dictionary object with the current :class:`SceneDict`.

        Parameter ``content`` (dict or :class:`~eradiate.scenes.base.SceneHelper`)
            Content to merge with the current scene. If ``content`` is a
            :class:`~eradiate.scenes.base.SceneHelper` instance, its
            :meth:`~eradiate.scenes.base.SceneHelper.kernel_dict` method will
            be called with ``ref`` set to `True`.

        Returns → ``self``
        """

        if isinstance(content, SceneHelper):
            for key, value in content.kernel_dict(ref=True).items():
                self[key] = value
        else:
            for key, value in content.items():
                self[key] = value

        return self

    def normalize(self):
        self["type"] = "scene"

    def load(self):
        self.check()
        from eradiate.kernel.core.xml import load_dict
        return load_dict(self)
