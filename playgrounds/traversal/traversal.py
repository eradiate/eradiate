from __future__ import annotations

import enum
import itertools
import typing as t
from collections import UserDict

import attrs
import mitsuba as mi
import numpy as np

# -- Utilities -----------------------------------------------------------------


def flatten(d: dict, sep: str = ".", name: str = "") -> dict:
    """
    Flatten a nested dictionary.

    Parameters
    ----------
    d : dict
        Dictionary to be flattened.

    name : str, optional, default: ""
        Path to the parent dictionary. By default, no parent name is defined.

    sep : str, optional, default: "."
        Flattened dict key separator.

    Returns
    -------
    dict
        A flattened copy of `d`.
    """
    result = {}

    for k, v in d.items():
        full_key = k if not name else f"{name}{sep}{k}"
        if isinstance(v, dict):
            result.update(flatten(v, sep=sep, name=full_key))
        else:
            result[full_key] = v

    return result


def set_nested(d: dict, path: str, value: t.Any, sep: str = "."):
    """
    Set values in a nested dictionary using a flat path.

    Parameters
    ----------
    d : dict
        Dictionary to operate on.

    path : str
        Path to the value to be set.

    value
        Value to which `path` is to be set.

    sep : str, optional, default: "."
        Separator used to decompose `path`.
    """
    *path, last = path.split(sep)
    for bit in path:
        d = d.setdefault(bit, {})
    d[last] = value


def unflatten(d: dict, sep: str = ".") -> dict:
    """
    Turn a flat dictionary into a nested dictionary.

    Parameters
    ----------
    d : dict
        Dictionary to be unflattened.

    sep : str, optional, default: "."
        Flattened dict key separator.

    Returns
    -------
    dict
        A nested copy of `d`.
    """
    result = {}

    for key, value in d.items():
        set_nested(result, key, value, sep)

    return result


# -- Kernel dict ---------------------------------------------------------------


class ParamFlags(enum.Flag):
    """
    Parameter flags.
    """

    SPECTRAL = enum.auto()
    GEOMETRIC = enum.auto()


@attrs.define
class Param:
    """
    A kernel scene parameter generator.
    """

    #: An attached callable which evaluates the parameter.
    _callable: t.Callable = attrs.field(repr=False)

    #: Flags specifying parameter attributes.
    flags: t.Optional[ParamFlags] = attrs.field(default=None)

    def __call__(self, *args, **kwargs):
        return self._callable(*args, **kwargs)


@attrs.define
class ParameterMap(UserDict):
    """
    A dict-like structure mapping parameter paths to methods generating them.
    """

    data: dict[str, Param] = attrs.field(factory=dict)

    def render(self, **kwargs) -> dict:
        """
        Evaluate the parameter map for a set of arguments.
        """
        return {key: param(**kwargs) for key, param in self.data.items()}


class _Empty(enum.Enum):
    """Sentinel value for empty kernel dictionary template entries."""

    EMPTY = enum.auto()

    def __repr__(self):
        return "EMPTY"

    def __bool__(self):
        return False


EMPTY = _Empty.EMPTY


@attrs.define
class KernelDictTemplate(UserDict):
    """
    A dictionary containing placeholders meant to be substituted using a
    :class:`ParameterMap`.
    """

    data: dict = attrs.field(factory=dict)

    def render(self, update_map: ParameterMap, **kwargs) -> dict:
        """
        Render the template as a nested dictionary using a parameter map to fill
        in empty fields.
        """
        result = self.data.copy()
        result.update(update_map.render(**kwargs))
        # TODO: check for leftover template values
        return unflatten(result, sep=".")


@attrs.define
class KernelDict:
    template: KernelDictTemplate = attrs.field(
        factory=KernelDictTemplate, converter=KernelDictTemplate
    )
    update_map: ParameterMap = attrs.field(factory=ParameterMap, converter=ParameterMap)

    def render(self, **kwargs):
        """
        Render the kernel dictionary template with the passed parameter map.
        """
        return self.template.render(self.update_map, **kwargs)


# -- Scene elements ------------------------------------------------------------


@attrs.define(eq=False)
class SceneElement:
    """
    Important: All subclasses *must* have a hash, thus eq must be False (see
    attrs docs on hashing for a complete explanation).
    """

    @property
    def kernel_type(self) -> t.Optional[str]:
        """
        Kernel type if this scene element can be modelled by a single kernel
        scene graph node; ``None`` otherwise. The default implementation raises
        a :class:`NotImplementedError`.
        """
        raise NotImplementedError

    @property
    def template(self) -> t.Mapping:
        """
        Kernel dictionary template contents associated with this scene element.
        The default implementation raises a :class:`NotImplementedError`.
        """
        return {} if self.kernel_type is None else {"type": self.kernel_type}

    @property
    def params(self) -> t.Optional[t.Mapping[str, Param]]:
        """
        Map of updatable parameters associated with this scene element.
        """
        return None

    @property
    def objects(self) -> t.Optional[t.Mapping[str, SceneElement]]:
        """
        Map of child objects associated with this scene element.
        """
        return None

    def traverse(self, callback: SceneTraversal) -> None:
        """
        Traverse this scene element and collect kernel dictionary template,
        parameter and object map contributions.

        Parameters
        ----------
        callback : SceneTraversal
            Callback data structure storing the collected data.
        """
        callback.put_template(self.template)

        if self.params is not None:
            callback.put_params(self.params)

        if self.objects is not None:
            if self.kernel_type is not None:
                for name, obj in self.objects.items():
                    callback.put_object(name, obj)
            else:
                for _, obj in self.objects.items():
                    obj.traverse(callback)


@attrs.define(eq=False)
class UniformSpectrum(SceneElement):
    value: float = attrs.field(default=1.0, converter=float)

    @property
    def kernel_type(self) -> str:
        return "uniform"

    @property
    def template(self) -> dict:
        return {**self.template, "value": self.value}


@attrs.define(eq=False)
class InterpolatedSpectrum(SceneElement):
    wavelengths: np.ndarray = attrs.field(converter=np.array)
    values: np.ndarray = attrs.field(converter=np.array)

    @property
    def kernel_type(self) -> t.Optional[str]:
        return "uniform"

    def eval(self, /, w):
        return np.interp(w, self.wavelengths, self.values)

    @property
    def template(self) -> t.Mapping:
        return {**super().template, "value": EMPTY}

    @property
    def params(self) -> t.Dict[str, Param]:
        return {"value": Param(self.eval, ParamFlags.SPECTRAL)}


@attrs.define(eq=False)
class DiffuseBSDF(SceneElement):
    reflectance: UniformSpectrum | InterpolatedSpectrum = attrs.field(
        factory=lambda: UniformSpectrum(0.5)
    )

    @property
    def kernel_type(self) -> str:
        return "diffuse"

    @property
    def objects(self) -> t.Dict[str, UniformSpectrum | InterpolatedSpectrum]:
        return {"reflectance": self.reflectance}


@attrs.define(eq=False)
class RectangleShape(SceneElement):
    bsdf: DiffuseBSDF = attrs.field(factory=DiffuseBSDF)

    @property
    def kernel_type(self) -> str:
        return "rectangle"

    @property
    def objects(self) -> t.Dict[str, DiffuseBSDF]:
        return {"bsdf": self.bsdf}


@attrs.define(eq=False)
class MultiShape(SceneElement):
    shapes: t.List[RectangleShape] = attrs.field(factory=list)

    @property
    def kernel_type(self) -> None:
        return None

    @property
    def objects(self) -> t.Dict[str, RectangleShape]:
        return {f"shape_{i}": shape for i, shape in enumerate(self.shapes)}


@attrs.define(eq=False)
class PathIntegrator(SceneElement):
    @property
    def kernel_type(self) -> str:
        return "path"


@attrs.define
class MonoSpectralConfig:
    wavelengths: list = attrs.field(factory=lambda: [440.0, 550.0, 660.0])


@attrs.define(eq=False)
class PerspectiveCamera(SceneElement):
    spectral_cfg = attrs.field(factory=MonoSpectralConfig)

    @property
    def kernel_type(self) -> str:
        return "perspective"

    @property
    def spectral_indexes(self) -> list:
        return self.spectral_cfg.wavelengths


@attrs.define(eq=False)
class Scene(SceneElement):
    _objects: dict = attrs.field(factory=dict)
    _partials: list = attrs.field(factory=list)

    @property
    def kernel_type(self) -> str:
        return "scene"

    @property
    def objects(self):
        return self._objects

    @property
    def partials(self):
        return self._partials


# -- Traversal -----------------------------------------------------------------


@attrs.define
class SceneTraversal:
    #: Current traversal node
    node: SceneElement

    #: Parent to current node
    parent: t.Optional[SceneElement] = attrs.field(default=None)

    #: Current node's name
    name: t.Optional[str] = attrs.field(default=None)

    #: Current depth
    depth: int = attrs.field(default=0)

    #: Dictionary mapping nodes to their parents
    hierarchy: dict = attrs.field(factory=dict)

    #: Kernel dictionary template
    template: dict = attrs.field(factory=dict)

    #: Dictionary mapping nodes to their defined parameters
    params: dict = attrs.field(factory=dict)

    def __attrs_post_init__(self):
        self.hierarchy[self.node] = (self.parent, self.depth)

    def put_template(self, template: dict):
        prefix = "" if self.name is None else f"{self.name}."

        for k, v in template.items():
            self.template[f"{prefix}{k}"] = v

    def put_params(self, params: dict):
        prefix = "" if self.name is None else f"{self.name}."

        for k, v in params.items():
            self.params[f"{prefix}{k}"] = v

    def put_object(self, name: str, node: SceneElement):
        if node is None or node in self.hierarchy:
            return

        cb = type(self)(
            node=node,
            parent=self.node,
            name=name if self.name is None else f"{self.name}.{name}",
            depth=self.depth + 1,
            hierarchy=self.hierarchy,
            template=self.template,
            params=self.params,
        )
        node.traverse(cb)


def traverse(node: SceneElement) -> KernelDict:
    # Traverse scene element tree
    cb = SceneTraversal(node)
    node.traverse(cb)

    # Use collected data to generate the kernel dictionary
    return KernelDict(cb.template, cb.params)


# -- Experiments ---------------------------------------------------------------


@attrs.define
class Experiment:
    kernel_scene: t.Optional["mitsuba.Scene"] = attrs.field(default=None, repr=False)
    results: dict = attrs.field(factory=dict)

    @property
    def measures(self) -> t.List[PerspectiveCamera]:
        raise NotImplementedError

    def kernel_dict(self) -> KernelDict:
        raise NotImplementedError

    def process(self, *measures, rebuild_kernel_scene: bool = False, spp=4):
        mi.set_variant("scalar_mono_double")

        if not measures:
            measures = self.measures

        self.results.clear()
        kernel_dict = self.kernel_dict()

        if self.kernel_scene is None or rebuild_kernel_scene:
            print("Loading kernel scene")
            self.kernel_scene = mi.load_dict(kernel_dict.render(w=500.0))
        kernel_scene_params = mi.traverse(self.kernel_scene)

        print("Parametric loop")
        spectral_indexes = sorted(
            set(itertools.chain(*(m.spectral_indexes for m in measures)))
        )

        for w in spectral_indexes:
            print(f"{w = }")
            print("Updating scene")
            update_map = kernel_dict.update_map.render(w=w)
            kernel_scene_params.update(update_map)

            for sensor in self.kernel_scene.sensors():
                loop_index = (w, sensor.id())

                self.results[loop_index] = mi.render(
                    self.kernel_scene, sensor=sensor, spp=spp
                )
                print()

        return self.results


@attrs.define
class SomeExperiment(Experiment):
    shape: RectangleShape = attrs.field(kw_only=True)
    integrator: PathIntegrator = attrs.field(factory=PathIntegrator)
    measures: t.List[PerspectiveCamera] = attrs.field(
        factory=PerspectiveCamera,
        converter=lambda x: [x] if isinstance(x, PerspectiveCamera) else x,
        validator=attrs.validators.deep_iterable(
            member_validator=attrs.validators.instance_of(PerspectiveCamera),
            iterable_validator=attrs.validators.instance_of(list),
        ),
    )

    def kernel_dict(self):
        return traverse(
            Scene(
                objects={
                    "shape": self.shape,
                    "integrator": self.integrator,
                    **{
                        f"camera_{i}": measure
                        for i, measure in enumerate(self.measures)
                    },
                }
            )
        )
