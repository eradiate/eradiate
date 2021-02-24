import os
from abc import ABC, abstractmethod
from copy import copy
from pathlib import Path

import attr
import pinttr
from matplotlib import pyplot as plt

import eradiate

from ._runner import runner
from ... import unit_registry as ureg
from ..._attrs import documented, parse_docs
from ..._mode import ModeNone
from ...exceptions import ModeError
from ...scenes.measure import DistantMeasure


@parse_docs
@attr.s
class SolverApp(ABC):
    """Abstract base class for solver applications."""

    _SCENE_TYPE = None  # Scene type; must be defined by child classes
    _raw_results = attr.ib(default=None, init=False, repr=False)  # Raw runner output

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
        """Create an instance after preparatory checks.
        All arguments are forwarded to the constructor.
        """
        if isinstance(eradiate.mode(), ModeNone):
            raise ModeError(f"no mode selected, use eradiate.set_mode()")

        return cls(*args, **kwargs)

    @classmethod
    def from_dict(cls, d):
        """Instantiate from a dictionary."""
        solver_config = pinttr.interpret_units(d, ureg=ureg)

        # Collect mode configuration
        try:
            # We make a copy because we'll mutate it later
            mode_config = copy(solver_config.pop("mode"))
        except KeyError:
            raise ValueError("section 'mode' missing from configuration dictionary")

        try:
            mode_id = mode_config.pop("type")
        except KeyError or AttributeError:
            raise ValueError(
                "parameter 'mode.type' missing from configuration dictionary"
            )

        # Select appropriate operational mode
        eradiate.set_mode(mode_id, **mode_config)
        if not eradiate.mode().is_monochromatic():
            raise ModeError("only monochromatic modes are supported")

        # Create scene
        scene = cls._SCENE_TYPE.from_dict(solver_config)

        # Instantiate class
        return cls.new(scene=scene)

    def process(self):
        """Run simulation on the configured scene. Raw results yielded by the
        encapsulated runner are stored in ``self._raw_results``.

        .. seealso:: :meth:`postprocess`, :meth:`run`
        """
        # Unset results for error detection
        self._raw_results = None
        self._results = dict()

        # Run simulation
        kernel_dict = self.scene.kernel_dict()
        self._raw_results = runner(kernel_dict)

    @abstractmethod
    def postprocess(self):
        """Post-process raw results stored in hidden attribute
        ``self._raw_results`` after successful execution of :meth:`process`.
        Post-processed results are stored in ``self.results``.

        Raises → ValueError:
            If ``self._raw_results`` is ``None``, *i.e.* if :meth:`process`
            has not been successfully run.

        .. seealso:: :meth:`process`, :meth:`run`
        """
        pass

    def run(self):
        """Perform radiative transfer simulation and post-process results.
        Essentially chains :meth:`process` and :meth:`postprocess`.
        """
        self.process()
        self.postprocess()

    def save_results(self, fname_prefix):
        """Save results to netCDF files.

        Parameter ``fname_prefix`` (str):
            Filename prefix for result storage. A netCDF file is created for
            each measure.
        """
        fname_prefix = Path(fname_prefix)
        for measure_id, results in self._results.items():
            fname_results = os.path.abspath(f"{fname_prefix}_{measure_id}.nc")
            os.makedirs(os.path.dirname(fname_results), exist_ok=True)
            print(f"Saving results to {fname_results}")
            results.to_netcdf(path=fname_results)

    def plot_results(self, fname_prefix):
        """Make default plots for stored results and save them to the hard
        drive.

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
