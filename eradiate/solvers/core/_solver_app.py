import os
from abc import ABC
from pathlib import Path

import attr
import numpy as np
import pinttr
from matplotlib import pyplot as plt

import eradiate
from ._runner import runner
from ..._attrs import documented, parse_docs
from ..._units import unit_context_kernel as uck, unit_registry as ureg
from ...exceptions import ModeError
from ...contexts import KernelDictContext
from ...scenes.measure._distant import DistantMeasure


@parse_docs
@attr.s
class SolverApp(ABC):
    """
    Abstract base class for solver applications.
    """

    _SCENE_TYPE = None  # Scene type; must be defined by child classes

    # Scene to simulate radiative transfer on; this field must be overridden by child classes
    # Type must match _SCENE_TYPE
    scene = attr.ib(default=None)

    # Post-processed output
    _results = documented(
        attr.ib(factory=dict, init=False, repr=False),
        doc="Post-processed simulation results. Each entry uses a measure ID as its "
        "key and holds a value consisting of a :class:`~xarray.Dataset` holding "
        "one variable per physical quantity computed by the measure.\n"
        "\n"
        ".. note:: This field is read-only and is not accessible as a "
        "constructor argument.",
        type="dict",
    )

    @property
    def results(self):
        return self._results

    @classmethod
    def new(cls, *args, **kwargs):
        """
        Create an instance after preparatory checks.
        All arguments are forwarded to the constructor.
        """
        if eradiate.mode() is None:
            raise ModeError("no mode selected, use eradiate.set_mode()")

        return cls(*args, **kwargs)

    @classmethod
    def from_dict(cls, d):
        """
        Instantiate from a dictionary.
        """
        solver_config = pinttr.interpret_units(d, ureg=ureg)

        # Collect mode configuration
        try:
            mode_id = solver_config.pop("mode")
        except KeyError:
            raise ValueError("parameter 'mode' missing from configuration dictionary")

        # Select appropriate operational mode
        eradiate.set_mode(mode_id)

        if not eradiate.mode().is_monochromatic():
            raise ModeError("only monochromatic modes are supported")

        # Create scene
        scene = cls._SCENE_TYPE.from_dict(solver_config)

        # Instantiate class
        return cls.new(scene=scene)

    def process(self, measure=None):
        """
        Run simulation on the configured scene. Raw results yielded by the
        encapsulated runner are stored in ``self._raw_results``.

        .. seealso:: :meth:`postprocess`, :meth:`run`

        Parameter ``measure`` (:class:`.Measure` or None):
            Measure for which to compute radiative transfer. If set to ``None``,
            the first element of ``self.measures`` is used.
        """
        # Select measure
        if measure is None:
            measure = self.scene.measures[0]

        # Unset results for error detection
        measure.raw_results = None

        # Create raw result storage
        ds = measure.raw_results_empty()

        # Spectral loop
        for spectral_ctx in measure.spectral_cfg.spectral_ctxs():
            # Initialise context
            ctx = KernelDictContext(spectral_ctx=spectral_ctx, ref=True)

            # Define sensor IDs
            sensor_ids = [sensor_info.id for sensor_info in measure.sensor_infos()]

            # Run simulation
            kernel_dict = self.scene.kernel_dict(ctx=ctx)
            run_results = runner(kernel_dict, sensor_ids)

            # Collect results
            for sensor_id, sensor_results in run_results.items():
                ds["raw_results"].loc[
                    dict(wavelength=spectral_ctx.wavelength, sensor_id=sensor_id)
                ] = sensor_results.squeeze()

        measure.raw_results = ds

    def postprocess(self, measure=None):
        """
        Post-process raw results stored in measures' ``raw_results`` field. This
        requires a successful execution of :meth:`process`. Post-processed results
        are stored in ``self.results``.

        Parameter ``measure`` (:class:`.Measure` or None):
            Measure for which to compute radiative transfer. If set to ``None``,
            the first element of ``self.measures`` is used.

        Raises → ValueError:
            If ``measure.raw_results`` is ``None``, *i.e.* if :meth:`process`
            has not been successfully run.

        .. seealso:: :meth:`process`, :meth:`run`
        """
        if not eradiate.mode().is_monochromatic():
            raise ModeError(f"unsupported mode '{eradiate.mode().id}'")

        # Select measure
        if measure is None:
            measure = self.scene.measures[0]

        # Collect measure results
        ds = measure.postprocessed_results()

        # Collect illumination data
        illumination = self.scene.illumination
        irradiance = illumination.irradiance.eval(spectral_ctx=measure.spectral_ctx)
        saa = illumination.azimuth.m_as(ureg.rad)
        sza = illumination.zenith.m_as(ureg.rad)
        cos_sza = np.cos(sza)

        # Add new dimensions
        ds = ds.expand_dims({"sza": [sza], "saa": [saa]}, axis=(0, 1))

        # Add illumination-dependent variables
        ds["irradiance"] = (
            ("sza", "saa", "wavelength"),
            np.array(irradiance.m_as(uck.get("irradiance")) * cos_sza).reshape(
                (1, 1, 1)
            ),
        )

        ds["brdf"] = ds["lo"] / ds["irradiance"]
        ds["brf"] = ds["brdf"] * np.pi

        self._results[measure.id] = ds

    def run(self, measure=None):
        """
        Perform radiative transfer simulation and post-process results.
        Essentially chains :meth:`process` and :meth:`postprocess`.

        Parameter ``measure`` (:class:`.Measure` or None):
            Measure for which to compute radiative transfer. If set to ``None``,
            the first element of ``self.measures`` is used.
        """
        self.process(measure)
        self.postprocess(measure)

    def save_results(self, fname_prefix):
        """
        Save results to NetCDF files.

        Parameter ``fname_prefix`` (str):
            Filename prefix for result storage. A NetCDF file is created for
            each measure.
        """
        fname_prefix = Path(fname_prefix)
        for measure_id, results in self._results.items():
            fname_results = os.path.abspath(f"{fname_prefix}_{measure_id}.nc")
            os.makedirs(os.path.dirname(fname_results), exist_ok=True)
            print(f"Saving results to {fname_results}")
            results.to_netcdf(path=fname_results)

    def plot_results(self, fname_prefix):
        """
        Make default plots for stored results and save them to the hard drive.

        Parameter ``fname_prefix`` (str):
            Filename prefix for plot files. A plot file is create for each
            computed quantity of each measure.
        """
        for measure in self.scene.measures:
            measure_id = measure.id
            result = self._results[measure_id]

            if isinstance(measure, DistantMeasure):
                for quantity in result.data_vars:
                    if quantity == "irradiance":
                        continue

                    data = result[quantity]

                    if data.shape[2] == 1:  # Only a plane is covered
                        data.squeeze().plot(x="vza")
                    else:
                        data.squeeze().plot.pcolormesh()

                    fname_plot = os.path.abspath(
                        f"{fname_prefix}_{measure_id}_{quantity}.png"
                    )
                    os.makedirs(os.path.dirname(fname_plot), exist_ok=True)
                    print(f"Saving plot to {fname_plot}")
                    plt.savefig(fname_plot, bbox_inches="tight")
                    plt.close()

            else:
                raise NotImplementedError
