import datetime

import attr
import numpy as np
import xarray as xr

import eradiate

from ._scene import RamiScene
from ..core._postprocess import distant_measure_compute_viewing_angles
from ..core._solver_app import SolverApp
from ... import unit_context_kernel as uck
from ... import unit_registry as ureg
from ..._attrs import documented, parse_docs
from ..._util import ensure_array
from ...scenes.measure._distant import DistantMeasure
from ...xarray.metadata import DatasetSpec, VarSpec


@parse_docs
@attr.s
class RamiSolverApp(SolverApp):
    """Solver application dedicated to the simulation of radiative transfer on
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

    def postprocess(self):
        """Post-process raw results stored in hidden attribute
        ``self._raw_results`` after successful execution of :meth:`process`.
        Post-processed results are stored in ``self.results``.

        Raises → ValueError:
            If ``self._raw_results`` is ``None``, *i.e.* if :meth:`process`
            has not been successfully run.

        .. seealso:: :meth:`process`, :meth:`run`
        """
        if self._raw_results is None:
            raise ValueError(
                f"raw results are unset: simulation must first be completed "
                f"using {self.__class__.__name__}.process()"
            )

        scene = self.scene

        # Ensure that scalar values used as xarray coordinates are arrays
        illumination = scene.illumination

        # Collect illumination parameters
        sza = ensure_array(illumination.zenith.m_as(ureg.deg), dtype=float)
        cos_sza = np.cos(illumination.zenith.m_as(ureg.rad))
        saa = ensure_array(illumination.azimuth.m_as(ureg.deg), dtype=float)
        wavelength = ensure_array(eradiate.mode().wavelength.magnitude, dtype=float)

        # Format results
        for measure in scene.measures:
            # Collect results from sensors associated to processed measure
            data = measure.postprocess_results(self._raw_results)

            if len(data.shape) != 2:
                raise ValueError(
                    f"raw result array has incorrect shape " f"{data.shape}"
                )

            # Add missing dimensions to raw result array
            data = np.expand_dims(data, [0, 1, -1])

            # Create an empty dataset to store results
            ds = xr.Dataset()

            if isinstance(measure, DistantMeasure):
                # Assign leaving radiance variable and corresponding coords
                ds["lo"] = xr.DataArray(
                    data,
                    coords=(
                        ("sza", sza),
                        ("saa", saa),
                        ("y", range(data.shape[2])),
                        ("x", range(data.shape[3])),
                        ("wavelength", wavelength),
                    ),
                )

                # Add viewing angles
                ds = distant_measure_compute_viewing_angles(ds, measure)

            else:
                raise ValueError(f"Unsupported measure type {measure.__class__}")

            # Add other variables
            irradiance = self.scene.illumination.irradiance.values
            ds["irradiance"] = (
                ("sza", "saa", "wavelength"),
                np.array(irradiance.m_as(uck.get("irradiance")) * cos_sza).reshape(
                    (1, 1, 1)
                ),
            )
            ds["brdf"] = ds["lo"] / ds["irradiance"]
            ds["brf"] = ds["brdf"] * np.pi

            # Add missing metadata
            dataset_spec = DatasetSpec(
                convention="CF-1.8",
                title="Top-of-canopy simulation results",
                history=f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
                f"data creation - {__name__}.{self.__class__.__name__}.postprocess",
                source=f"eradiate, version {eradiate.__version__}",
                references="",
                var_specs={
                    "irradiance": VarSpec(
                        standard_name="toc_horizontal_solar_irradiance_per_unit_wavelength",
                        units=str(uck.get("irradiance")),
                        long_name="top-of-canopy horizontal spectral irradiance",
                    ),
                    "lo": VarSpec(
                        standard_name="toc_outgoing_radiance_per_unit_wavelength",
                        units=str(uck.get("radiance")),
                        long_name="top-of-canopy outgoing spectral radiance",
                    ),
                    "brf": VarSpec(
                        standard_name="toc_brf",
                        units="dimensionless",
                        long_name="top-of-canopy bi-directional reflectance factor",
                    ),
                    "brdf": VarSpec(
                        standard_name="toc_brdf",
                        units="1/sr",
                        long_name="top-of-canopy bi-directional reflection distribution function",
                    ),
                },
                coord_specs="angular_observation",
            )
            ds.ert.normalize_metadata(dataset_spec)

            self._results[measure.id] = ds
