import datetime

import attr

import eradiate

from ._scene import RamiScene
from ..core._solver_app import SolverApp
from ... import unit_context_kernel as uck
from ...attrs import documented, parse_docs
from ...units import symbol
from ...xarray.metadata import DatasetSpec, VarSpec


@parse_docs
@attr.s
class RamiSolverApp(SolverApp):
    """
    Solver application dedicated to the simulation of radiative transfer on
    RAMI benchmark scenes.
    """

    _SCENE_TYPE = RamiScene

    scene = documented(
        attr.ib(
            factory=RamiScene,
            converter=lambda x: RamiScene.from_dict(x) if isinstance(x, dict) else x,
            validator=attr.validators.instance_of(RamiScene),
        ),
        doc="RAMI benchmark scene to simulate radiative transfer on. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by :meth:`RamiScene.from_dict() <.RamiScene.from_dict>`.",
        type=":class:`RamiScene`",
        default=":class:`RamiScene() <RamiScene>`",
    )

    def postprocess(self, measure=None):
        # Select measure
        if measure is None:
            measure = self.scene.measures[0]

        # Apply post-processing
        super(RamiSolverApp, self).postprocess(measure)

        # Add missing metadata
        ds = self.results[measure.id]

        dataset_spec = DatasetSpec(
            convention="CF-1.8",
            title="Top-of-canopy simulation results",
            history=f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
            f"data creation - {__name__}.{self.__class__.__name__}.postprocess",
            source=f"eradiate, version {eradiate.__version__}",
            references="",
        )
        ds.ert.normalize_metadata(dataset_spec)
