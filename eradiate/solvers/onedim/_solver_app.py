import datetime

import attr

import eradiate
from ._scene import OneDimScene
from ..core._solver_app import SolverApp
from ... import unit_context_kernel as uck
from ..._attrs import documented, parse_docs
from ...xarray.metadata import DatasetSpec, VarSpec


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
            converter=lambda x: OneDimScene.from_dict(x) if isinstance(x, dict) else x,
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
            var_specs={
                "irradiance": VarSpec(
                    standard_name="toa_horizontal_solar_irradiance_per_unit_wavelength",
                    units=str(uck.get("irradiance")),
                    long_name="top-of-atmosphere horizontal spectral irradiance",
                ),
                "lo": VarSpec(
                    standard_name="toa_outgoing_radiance_per_unit_wavelength",
                    units=str(uck.get("radiance")),
                    long_name="top-of-atmosphere outgoing spectral radiance",
                ),
                "brf": VarSpec(
                    standard_name="toa_brf",
                    units="dimensionless",
                    long_name="top-of-atmosphere bi-directional reflectance factor",
                ),
                "brdf": VarSpec(
                    standard_name="toa_brdf",
                    units="1/sr",
                    long_name="top-of-atmosphere bi-directional reflection distribution function",
                ),
            },
            coord_specs="angular_observation",
        )
        ds.ert.normalize_metadata(dataset_spec)
