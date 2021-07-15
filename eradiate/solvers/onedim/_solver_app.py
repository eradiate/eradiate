import datetime
from collections import MutableMapping

import attr

import eradiate

from ._scene import OneDimScene
from ..core._solver_app import SolverApp
from ...attrs import documented, parse_docs
from ...xarray.metadata import DatasetSpec


@parse_docs
@attr.s
class OneDimSolverApp(SolverApp):
    """
    Solver application dedicated to the simulation of radiative transfer on
    one-dimensional scenes.
    """

    _SCENE_TYPE = OneDimScene

    scene = documented(
        attr.ib(
            factory=OneDimScene,
            converter=lambda x: OneDimScene(**x)
            if isinstance(x, MutableMapping)
            else x,
            validator=attr.validators.instance_of(OneDimScene),
        ),
        doc="One-dimensional scene to simulate radiative transfer on. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by "
        ":meth:`OneDimScene.from_dict() <.OneDimScene.from_dict>`.",
        type=":class:`OneDimScene`",
        default=":class:`OneDimScene() <.OneDimScene>`",
    )

    def postprocess(self, measure=None):
        # Select measure
        if measure is None:
            measure = self.scene.measures[0]

        # Apply post-processing
        super(OneDimSolverApp, self).postprocess(measure)

        # Add missing metadata
        ds = self.results[measure.id]

        dataset_spec = DatasetSpec(
            convention="CF-1.8",
            title="Top-of-atmosphere simulation results",
            history=f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
            f"data creation - {__name__}.{self.__class__.__name__}.postprocess",
            source=f"eradiate, version {eradiate.__version__}",
            references="",
        )
        ds.ert.normalize_metadata(dataset_spec)
