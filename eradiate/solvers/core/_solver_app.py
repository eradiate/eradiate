import logging
import os
import typing as t
from abc import ABC
from pathlib import Path

import attr
import pinttr
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

import eradiate

from ._runner import runner
from ..._config import config
from ..._mode import ModeFlags
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext
from ...exceptions import ModeError, UnsupportedModeError
from ...scenes.measure import Measure, MeasureResults
from ...scenes.measure._distant import (
    DistantAlbedoMeasure,
    DistantRadianceMeasure,
    DistantReflectanceMeasure,
)
from ...units import unit_registry as ureg

logger = logging.getLogger(__name__)


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

        if not eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            raise UnsupportedModeError(supported="monochromatic")

        # Create scene
        scene = cls._SCENE_TYPE.from_dict(solver_config)

        # Instantiate class
        return cls.new(scene=scene)

    def process(self, *measures: t.Union[Measure, int]):
        """
        Run simulation on the configured scene. Raw results yielded by the
        runner function are stored in ``measure.results``.

        .. seealso:: :meth:`postprocess`, :meth:`run`

        Parameter ``*measures`` (:class:`.Measure` or int):
            One or several measures for which to compute radiative transfer.
            Alternatively, indexes in the measure array can be passed.
            If no value is passed, all measures are processed.
        """
        # Mode safeguard
        if not eradiate.mode().has_flags(ModeFlags.ANY_MONO | ModeFlags.ANY_CKD):
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

        if not measures:
            measures = self.scene.measures

        for measure in measures:
            if isinstance(measure, int):
                measure = self.scene.measures[measure]

            logger.info(f"Processing measure '{measure.id}'")

            # Reset measure results
            measure.results = MeasureResults()

            # Spectral loop
            spectral_ctxs = measure.spectral_cfg.spectral_ctxs()

            with tqdm(
                initial=0,
                total=len(spectral_ctxs),
                unit_scale=1.0,
                leave=True,
                bar_format="{desc}{n_fmt}/{total_fmt}|{bar}| {elapsed}, ETA={remaining}",
                disable=config.progress < 1,
            ) as pbar:
                for spectral_ctx in spectral_ctxs:
                    pbar.set_description(
                        f"Spectral loop [{spectral_ctx.spectral_index_formatted}]",
                        refresh=True,
                    )

                    # Initialise context
                    ctx = KernelDictContext(spectral_ctx=spectral_ctx, ref=True)

                    # Collect sensor IDs
                    sensor_ids = [
                        sensor_info.id for sensor_info in measure.sensor_infos()
                    ]

                    # Run simulation
                    kernel_dict = self.scene.kernel_dict(ctx=ctx)
                    run_results = runner(kernel_dict, sensor_ids)

                    # Store results
                    measure.results.raw[spectral_ctx.spectral_index] = run_results

                    # Update progress display
                    pbar.update()

    def postprocess(self, *measures: t.Union[Measure, int]):
        """
        Post-process raw results stored in a measure's ``results`` field. This
        requires a successful execution of :meth:`process`. Post-processed results
        are stored in ``self.results``.

        Parameter ``*measure`` (:class:`.Measure` or int):
            One or several measures for which to perform post-processing.
            Alternatively, indexes in the measure array can be passed.
            If no value is passed, all measures are processed.

        Raises → ValueError:
            If ``measure.raw_results`` is ``None``, *i.e.* if :meth:`process`
            has not been successfully run.

        .. seealso:: :meth:`process`, :meth:`run`
        """
        if not eradiate.mode().has_flags(ModeFlags.ANY_MONO | ModeFlags.ANY_CKD):
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

        if not measures:
            measures = self.scene.measures

        for measure in measures:
            if isinstance(measure, int):
                measure = self.scene.measures[measure]

            # Prepare measure postprocessing arguments
            measure_kwargs = {}
            if isinstance(measure, (DistantReflectanceMeasure, DistantAlbedoMeasure)):
                measure_kwargs["illumination"] = self.scene.illumination

            # Collect measure results
            self._results[measure.id] = measure.postprocess(**measure_kwargs)

    def run(self, *measures: t.Union[Measure, int]):
        """
        Perform radiative transfer simulation and post-process results.
        Essentially chains :meth:`process` and :meth:`postprocess`.

        Parameter ``*measures`` (:class:`.Measure` or int):
            One or several measures for which to compute radiative transfer.
            Alternatively, indexes in the measure array can be passed.
            If no value is passed, all measures are processed.
        """
        self.process(*measures)
        self.postprocess(*measures)

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

            if isinstance(measure, DistantRadianceMeasure):
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
