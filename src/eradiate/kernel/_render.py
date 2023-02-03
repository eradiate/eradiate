import typing as t

import attrs
import mitsuba as mi
from tqdm.auto import tqdm

from . import UpdateMapTemplate
from .._config import ProgressLevel, config
from ..contexts import KernelDictContext
from ..rng import SeedState, root_seed_state


@attrs.define(repr=False)
class MitsubaScene:
    """
    This container aggregates a Mitsuba scene, its associated parameters and a
    set of updaters that can be used to modify the scene parameters.
    """

    scene: "mitsuba.Scene" = attrs.field()
    parameters: t.Optional["mitsuba.SceneParameters"] = attrs.field(default=None)
    umap_template: t.Optional[UpdateMapTemplate] = attrs.field(default=None)


def mi_traverse(
    scene: "mitsuba.Scene", umap_template: t.Optional[UpdateMapTemplate] = None
) -> MitsubaScene:
    """
    Traverse a node of Mitsuba's scene graph and return a dictionary-like
    object that can be used to read and write associated scene parameters.

    Notes
    -----
    This is a reimplementation of the :func:`mitsuba.traverse` function.
    """

    umap_template = (
        UpdateMapTemplate(data=umap_template.data.copy())
        if umap_template is not None
        else UpdateMapTemplate()
    )

    lookups = {
        k: v
        for k, v in umap_template
        if v.parameter_id is None and v.lookup is not None
    }

    class SceneTraversal(mi.TraversalCallback):
        def __init__(
            self,
            node,
            parent=None,
            properties=None,
            hierarchy=None,
            prefixes=None,
            name=None,
            depth=0,
            flags=+mi.ParamFlags.Differentiable,
        ):
            mi.TraversalCallback.__init__(self)
            self.properties = dict() if properties is None else properties
            self.hierarchy = dict() if hierarchy is None else hierarchy
            self.prefixes = set() if prefixes is None else prefixes

            if name is not None:
                ctr, name_len = 1, len(name)
                while name in self.prefixes:
                    name = "%s_%i" % (name[:name_len], ctr)
                    ctr += 1
                self.prefixes.add(name)

            self.name = name
            self.node = node
            self.depth = depth
            self.hierarchy[node] = (parent, depth)
            self.flags = flags

            # Try and recover a parameter ID from this node
            for uparam in list(lookups.values()):
                lookup_result = uparam.lookup(self.node, self.name)
                if lookup_result is not None:
                    uparam.parameter_id = lookup_result

        def put_parameter(self, name, ptr, flags, cpptype=None):
            name = name if self.name is None else self.name + "." + name

            flags = self.flags | flags
            # Non differentiable parameters shouldn't be flagged as discontinuous
            if (flags & mi.ParamFlags.NonDifferentiable) != 0:
                flags = flags & ~mi.ParamFlags.Discontinuous

            self.properties[name] = (ptr, cpptype, self.node, self.flags | flags)

        def put_object(self, name, node, flags):
            if node is None or node in self.hierarchy:
                return
            cb = SceneTraversal(
                node=node,
                parent=self.node,
                properties=self.properties,
                hierarchy=self.hierarchy,
                prefixes=self.prefixes,
                name=name if self.name is None else f"{self.name}.{name}",
                depth=self.depth + 1,
                flags=self.flags | flags,
            )
            node.traverse(cb)

    cb = SceneTraversal(scene)
    scene.traverse(cb)

    return MitsubaScene(
        scene=scene,
        parameters=mi.SceneParameters(cb.properties, cb.hierarchy),
        umap_template=umap_template,
    )


def mi_render(
    mi_scene: MitsubaScene,
    ctxs: t.List[KernelDictContext],
    sensors: t.Union[None, int, t.List[int]] = 0,
    spp: int = 0,
    seed_state: t.Optional[SeedState] = None,
) -> t.Dict[t.Any, "mitsuba.Bitmap"]:
    """
    Render a Mitsuba scene multiple times given specified contexts and sensor
    indices.

    Parameters
    ----------
    mi_scene : .MitsubaScene
        Mitsuba scene to render.

    ctxs : list of :class:`.KernelDictContext`
        List of contexts used to generate the parameter update table at each
        iteration.

    sensors : int or list of int, optional
        Sensor indices to render. If ``None``, all sensors are rendered.

    spp : int, optional, default: 0
        Number of samples per pixel. If set to 0, the value set in the original
        scene definition takes precedence.

    seed_state : :class:`.SeedState, optional
        Seed state used to generate seeds to initialise Mitsuba's RNG at
        each run. If unset, Eradiate's root seed state is used.

    Returns
    -------
    dict
        A nested dictionary mapping context and sensor indices to rendered
        bitmaps.
    """

    if seed_state is None:
        seed_state = root_seed_state

    results = {}

    # Loop on contexts
    with tqdm(
        initial=0,
        total=len(ctxs),
        unit_scale=1.0,
        leave=True,
        bar_format="{desc}{n:g}/{total:g}|{bar}| {elapsed}, ETA={remaining}",
        disable=(config.progress < ProgressLevel.SPECTRAL_LOOP) or len(ctxs) <= 1,
    ) as pbar:
        for ctx in ctxs:
            pbar.set_description(
                f"Eradiate [{ctx.index_formatted}]",
                refresh=True,
            )

            mi_scene.parameters.update(mi_scene.umap_template.render(ctx))

            if sensors is None:
                mi_sensors = [
                    (i, sensor) for i, sensor in enumerate(mi_scene.scene.sensors())
                ]

            else:
                if isinstance(sensors, int):
                    sensors = [sensors]
                mi_sensors = [(i, mi_scene.scene.sensors()[i]) for i in sensors]

            # Loop on sensors
            for i_sensor, mi_sensor in mi_sensors:
                # Render sensor
                mi.render(
                    mi_scene.scene,
                    sensor=i_sensor,
                    seed=int(seed_state.next()),
                    spp=spp,
                )

                # Store result in a new Bitmap object
                if ctx.spectral_ctx.spectral_index not in results:
                    results[ctx.spectral_ctx.spectral_index] = {}

                results[ctx.spectral_ctx.spectral_index][mi_sensor.id()] = mi.Bitmap(
                    mi_sensor.film().bitmap()
                )

            pbar.update()

    return results
