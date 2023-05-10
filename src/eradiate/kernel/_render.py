from __future__ import annotations

import logging
import typing as t
import warnings

import attrs
import mitsuba as mi
from tqdm.auto import tqdm

from ._kernel_dict import UpdateMapTemplate
from .._config import ProgressLevel, config
from ..attrs import documented, parse_docs
from ..contexts import KernelContext
from ..rng import SeedState, root_seed_state

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
#                         Parameter lookup strategies
# ------------------------------------------------------------------------------


@parse_docs
@attrs.frozen
class TypeIdLookupStrategy:
    """
    This parameter ID lookup strategy searches for a Mitsuba type and object ID
    match.

    Instances are callables which take, as argument, the current node during
    a Mitsuba scene tree traversal and, optionally, its path in the Mitsuba
    scene tree. If the lookup succeeds, the full parameter path is returned.
    """

    node_type: type = documented(
        attrs.field(validator=attrs.validators.instance_of(type)),
        doc="Type of the node which will be looked up.",
        type="type",
    )

    node_id: str = documented(
        attrs.field(validator=attrs.validators.instance_of(str)),
        doc="ID of the node which will be looked up.",
        type="str",
    )

    parameter_relpath: str = documented(
        attrs.field(validator=attrs.validators.instance_of(str)),
        doc="Parameter path relative to its parent object.",
        type="str",
    )

    def __call__(self, node, node_path: str | None = None) -> str | None:
        if isinstance(node, self.node_type) and node.id() == self.node_id:
            prefix = f"{node_path}." if node_path is not None else ""
            return f"{prefix}{self.parameter_relpath}"

        else:
            return None


# ------------------------------------------------------------------------------
#                           Mitsuba scene traversal
# ------------------------------------------------------------------------------


@parse_docs
@attrs.define
class MitsubaObjectWrapper:
    """
    This container aggregates a Mitsuba object, its associated parameters and a
    set of updaters that can be used to modify the scene parameters.

    See Also
    --------
    :func:`mi_traverse`
    """

    obj: mi.Object = documented(
        attrs.field(repr=lambda x: "Scene[...]" if isinstance(x, mi.Scene) else str(x)),
        doc="Mitsuba object.",
        type="mitsuba.Object",
    )

    parameters: mi.SceneParameters | None = documented(
        attrs.field(
            default=None,
            repr=lambda x: "SceneParameters[...]"
            if x.__class__.__name__ == "SceneParameters"
            else str(None),
        ),
        doc="Mitsuba scene parameter map.",
        type="mitsuba.SceneParameters",
        init_type="mitsuba.SceneParameters, optional",
        default="None",
    )

    umap_template: UpdateMapTemplate | None = documented(
        attrs.field(
            default=None,
            repr=lambda x: "UpdateMapTemplate[...]"
            if isinstance(x, UpdateMapTemplate)
            else str(x),
        ),
        doc="An update map template, which can be rendered and used to update "
        "Mitsuba scene parameters depending on context information.",
        type=".UpdateMapTemplate",
        init_type=".UpdateMapTemplate, optional",
        default="None",
    )

    def drop_parameters(self) -> None:
        """
        Reduce the size of the scene parameter table :attr:`.parameters` by
        only keeping elements whose keys are listed in the parameter update
        map template :attr:`.umap_template`. For parameters associated with a
        lookup protocol, the looked up parameter ID is checked and used.
        """
        if self.umap_template is not None:
            keys = []
            for name, param in self.umap_template.items():
                if param.lookup_strategy is not None:
                    if param.parameter_id is not None:
                        keys.append(param.parameter_id)
                    else:
                        warnings.warn(
                            f"Parameter '{name}' has a lookup strategy but the "
                            "associated parameter ID is undefined; was a "
                            "parameter lookup performed during the Mitsuba "
                            "scene traversal?"
                        )
                else:
                    keys.append(name)

            self.parameters.keep(keys)


def mi_traverse(
    obj: mi.Object,
    umap_template: UpdateMapTemplate | None = None,
) -> MitsubaObjectWrapper:
    """
    Traverse a node of the Mitsuba scene graph and return a container holding
    the Mitsuba scene, its parameter map and an updated parameter update map.

    Parameters
    ----------
    obj : mitsuba.Object
        Mitsuba scene graph node to be traversed.

    umap_template : .UpdateMapTemplate, optional
        An additional update map template which is to be updated during
        traversal. This is used to perform parameter lookup during traversal.

    Returns
    -------
    MitsubaObjectWrapper
        A container holding the traversed object, the corresponding parameter
        map and the parameter update map (if any).

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
        for k, v in umap_template.items()
        if v.parameter_id is None and v.lookup_strategy is not None
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
            for name, uparam in list(lookups.items()):
                lookup_result = uparam.lookup_strategy(self.node, self.name)
                if lookup_result is not None:
                    uparam.parameter_id = lookup_result
                    del lookups[
                        name
                    ]  # Remove successful lookups to accelerate future searches

        def put_parameter(self, name, ptr, flags, cpptype=None):
            name = name if self.name is None else self.name + "." + name

            flags = self.flags | flags
            # Non-differentiable parameters shouldn't be flagged as discontinuous
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

    cb = SceneTraversal(obj)
    obj.traverse(cb)

    # Check if there are unsuccessful lookups
    if lookups:
        warnings.warn(
            "There were unsuccessful Mitsuba scene parameter lookups: "
            f"{list(lookups.keys())}"
        )

    return MitsubaObjectWrapper(
        obj=obj,
        parameters=mi.SceneParameters(cb.properties, cb.hierarchy),
        umap_template=umap_template,
    )


# ------------------------------------------------------------------------------
#                             Mitsuba scene render
# ------------------------------------------------------------------------------


def mi_render(
    mi_scene: MitsubaObjectWrapper,
    ctxs: list[KernelContext],
    sensors: None | int | list[int] = None,
    spp: int = 0,
    seed_state: SeedState | None = None,
) -> dict[t.Any, mi.Bitmap]:
    """
    Render a Mitsuba scene multiple times given specified contexts and sensor
    indices.

    Parameters
    ----------
    mi_scene : .MitsubaObjectWrapper
        Mitsuba scene to render.

    ctxs : list of :class:`.KernelContext`
        List of contexts used to generate the parameter update table at each
        iteration.

    sensors : int or list of int, optional
        Sensor indices to render. If ``None`` (default), all sensors are
        rendered.

    spp : int, optional, default: 0
        Number of samples per pixel. If set to 0 (default), the value set in the
        original scene definition takes precedence.

    seed_state : .SeedState, optional
        Seed state used to generate seeds to initialize Mitsuba's RNG at
        each run. If unset, Eradiate's root seed state is used.

    Returns
    -------
    dict
        A nested dictionary mapping context and sensor indices to rendered
        bitmaps.

    Notes
    -----
    This function wraps sequential calls to  :func:`mitsuba.render`.
    """

    if seed_state is None:
        logger.debug("Using default RNG seed generator")
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

            logger.debug("Updating Mitsuba scene parameters")
            mi_scene.parameters.update(mi_scene.umap_template.render(ctx))

            if sensors is None:
                mi_sensors = [
                    (i, sensor) for i, sensor in enumerate(mi_scene.obj.sensors())
                ]

            else:
                if isinstance(sensors, int):
                    sensors = [sensors]
                mi_sensors = [(i, mi_scene.obj.sensors()[i]) for i in sensors]

            # Loop on sensors
            for i_sensor, mi_sensor in mi_sensors:
                # Render sensor
                seed = int(seed_state.next())
                logger.debug(
                    'Running Mitsuba for sensor "%s" with seed value %s',
                    mi_sensor.id(),
                    seed,
                )
                mi.render(mi_scene.obj, sensor=i_sensor, seed=seed, spp=spp)

                # Store result in a new Bitmap object
                siah = ctx.si.as_hashable
                if siah not in results:
                    results[siah] = {}

                results[siah][mi_sensor.id()] = mi.Bitmap(mi_sensor.film().bitmap())

            pbar.update()

    return results
