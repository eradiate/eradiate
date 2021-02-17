import datetime
import os
from pathlib import Path

import attr
import matplotlib.pyplot as plt
import numpy as np
import pinttr
import xarray as xr
from tinydb import (
    Query,
    TinyDB
)
from tinydb.storages import MemoryStorage

import eradiate.kernel

from ..onedim.runner import OneDimRunner
from ..._attrs import (
    documented,
    parse_docs
)
from ..._mode import ModeNone
from ..._units import unit_context_kernel as uck
from ..._units import unit_registry as ureg
from ..._util import ensure_array
from ...exceptions import ModeError
from ...frame import direction_to_angles
from ...scenes.biosphere import (
    BiosphereFactory,
    Canopy
)
from ...scenes.core import (
    KernelDict,
    SceneElement
)
from ...scenes.illumination import (
    DirectionalIllumination,
    IlluminationFactory
)
from ...scenes.integrators import (
    Integrator,
    IntegratorFactory,
    PathIntegrator
)
from ...scenes.measure import (
    DistantMeasure,
    MeasureFactory
)
from ...scenes.surface import (
    LambertianSurface,
    Surface,
    SurfaceFactory
)
from ...validators import is_positive
from ...warp import square_to_uniform_hemisphere
from ...xarray.metadata import (
    DatasetSpec,
    VarSpec
)


@parse_docs
@attr.s
class RamiScene(SceneElement):
    """
    Scene abstraction suitable for radiative transfer simulation on RAMI
    benchmark scenes.
    """
    surface = documented(
        attr.ib(
            factory=LambertianSurface,
            converter=SurfaceFactory.convert,
            validator=attr.validators.instance_of(Surface)
        ),
        doc="Surface specification. "
            "This parameter can be specified as a dictionary which will be "
            "interpreted by "
            ":meth:`SurfaceFactory.convert() <.SurfaceFactory.convert>`. "
            ".. note:: Surface size will be overridden using canopy parameters.",
        type=":class:`.Surface` or dict",
        default=":class:`LambertianSurface() <.LambertianSurface>`"
    )

    canopy = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(BiosphereFactory.convert),
            validator=attr.validators.optional(attr.validators.instance_of(Canopy))
        ),
        doc="Canopy specification. "
            "This parameter can be specified as a dictionary which will be "
            "interpreted by "
            ":meth:`BiosphereFactory.convert() <.BiosphereFactory.convert>`.",
        type=":class:`.Canopy` or dict",
        default=":class:`HomogeneousDiscreteCanopy() <.HomogeneousDiscreteCanopy>`",
    )

    padding = documented(
        attr.ib(
            default=0,
            converter=int,
            validator=is_positive
        ),
        doc="Padding level. The scene will be padded with copies to account for "
            "adjacency effects. This, in practice, has effects similar to "
            "making the scene periodic."
            "A value of 0 will yield only the defined scene. A value of 1 "
            "will add one copy in every direction, yielding a 3×3 patch. A "
            "value of 2 will yield a 5×5 patch, etc. The optimal padding level "
            "depends on the scene.",
        type="int",
        default="0"
    )

    illumination = documented(
        attr.ib(
            factory=DirectionalIllumination,
            converter=IlluminationFactory.convert,
            validator=attr.validators.instance_of(DirectionalIllumination)
        ),
        doc="Illumination specification. "
            "This parameter can be specified as a dictionary which will be "
            "interpreted by "
            ":meth:`IlluminationFactory.convert() <.IlluminationFactory.convert>`.",
        type=":class:`.DirectionalIllumination` or dict",
        default=":class:`DirectionalIllumination() <.DirectionalIllumination>`"
    )

    measures = documented(
        attr.ib(
            factory=lambda: [DistantMeasure()],
            converter=lambda value:
            [MeasureFactory.convert(x) for x in pinttr.util.always_iterable(value)]
            if not isinstance(value, dict)
            else [MeasureFactory.convert(value)]
        ),
        doc="List of measure specifications. The passed list may contain "
            "dictionary, which will be interpreted by "
            ":meth:`MeasureFactory.convert() <.MeasureFactory.convert>`. "
            "Optionally, a single :class:`.Measure` or dictionary specification "
            "may be passed and will automatically be wrapped into a list.\n"
            "\n"
            "Allowed value types: :class:`DistantMeasure`.\n"
            "\n"
            ".. note:: The target zone will be overridden using canopy "
            "parameters if unset. If no canopy is specified, surface size "
            "parameters will be used.",
        type="list[:class:`.Measure`] or list[dict] or :class:`.Measure` or dict",
        default=":class:`DistantMeasure() <.DistantMeasure>`",
    )

    @measures.validator
    def _measures_validator(self, attribute, value):
        for element in value:
            # Check measure type
            if not isinstance(element, DistantMeasure):
                raise TypeError(
                    f"while validating {attribute.name}: must be a list of "
                    f"objects of one of the following types: "
                    f"(DistantMeasure)"
                )

    integrator = documented(
        attr.ib(
            factory=PathIntegrator,
            converter=IntegratorFactory.convert,
            validator=attr.validators.instance_of(Integrator)
        ),
        doc="Monte Carlo integration algorithm specification. "
            "This parameter can be specified as a dictionary which will be "
            "interpreted by "
            ":meth:`IntegratorFactory.convert() <.IntegratorFactory.convert>`.",
        type=":class:`.Integrator` or dict",
        default=":class:`PathIntegrator() <.PathIntegrator>`",
    )

    measure_registry = attr.ib(
        init=False,
        repr=False,
        factory=lambda: TinyDB(storage=MemoryStorage)
    )

    def __attrs_post_init__(self):
        # Parts of the init sequence we could take care of using converters

        # Override surface width with canopy width
        if self.canopy is not None:
            self.surface.width = max(self.canopy.size[:2])

        # scale surface to accomodate padding
        self.surface.width = self.surface.width * (1 + 2 * self.padding)

        # Process measures
        for measure in self.measures:
            # Override ray target location if relevant
            if isinstance(measure, DistantMeasure):
                if measure.target is None:
                    if self.canopy is not None:
                        measure.target = dict(
                            type="rectangle",
                            xmin=-0.5 * self.canopy.size[0],
                            xmax=0.5 * self.canopy.size[0],
                            ymin=-0.5 * self.canopy.size[1],
                            ymax=0.5 * self.canopy.size[1],
                        )
                    else:
                        measure.target = dict(
                            type="rectangle",
                            xmin=-0.5 * self.surface.width,
                            xmax=0.5 * self.surface.width,
                            ymin=-0.5 * self.surface.width,
                            ymax=0.5 * self.surface.width,
                        )

        # Populate measure registry
        for measure in self.measures:
            for sensor_id, sensor_spp in measure.sensor_info():
                self.measure_registry.insert({
                    "measure_id": measure.id,
                    "sensor_id": sensor_id,
                    "sensor_spp": sensor_spp
                })

    def kernel_dict(self, ref=True):
        from eradiate.kernel.core import ScalarTransform4f
        result = KernelDict.empty()

        if self.canopy is not None:
            canopy_dict = self.canopy.kernel_dict(ref=True)
            result.add(canopy_dict)

            # specify the instances for padding
            patch_size = max(self.canopy.size[:2])
            kdu_length = uck.get("length")
            for shapegroup_id in canopy_dict.keys():
                if shapegroup_id.find("bsdf") != -1:
                    continue
                for x_offset in np.arange(-self.padding, self.padding + 1):
                    for y_offset in np.arange(-self.padding, self.padding + 1):
                        instance_dict = {
                            "type": "instance",
                            "group": {
                                "type": "ref",
                                "id": f"{shapegroup_id}"
                            },
                            "to_world": ScalarTransform4f.translate([
                                patch_size.m_as(kdu_length) * x_offset,
                                patch_size.m_as(kdu_length) * y_offset,
                                0.0
                            ]
                            )

                        }
                        result[f"instance{x_offset}_{y_offset}"] = instance_dict

        result.add([
            self.surface,
            self.illumination,
            *self.measures,
            self.integrator
        ])

        return result


@parse_docs
@attr.s
class RamiSolverApp:
    """
    Solver application dedicated to the simulation of radiative transfer on
    RAMI benchmark scenes.
    """
    # Instance attributes
    scene = documented(
        attr.ib(
            factory=RamiScene,
            converter=lambda x: RamiScene.from_dict(x)
            if isinstance(x, dict) else x,
            validator=attr.validators.instance_of(RamiScene)
        ),
        doc="RAMI benchmark scene to simulate radiative transfer on. "
            "This parameter can be specified as a dictionary which will be "
            "interpreted by :meth:`RamiScene.from_dict() <.RamiScene.from_dict>`.",
        type=":class:`RamiScene`",
        default=":class:`RamiScene() <RamiScene>`",
    )

    results = documented(
        attr.ib(factory=dict, init=False),
        doc="Post-processed simulation results. Each entry uses a measure ID as its "
            "key and holds a value consisting of a :class:`~xarray.Dataset` holding "
            "one variable per physical quantity computed by the measure.\n"
            "\n"
            ".. note:: This field is read-only and is not accessible as a "
            "constructor argument.",
        type="dict",
    )

    _kernel_dict = attr.ib(default=None, init=False, repr=False)  # Cached kernel dictionary

    _runner = attr.ib(default=None, init=False, repr=False)  # Runner

    _raw_results = attr.ib(default=None, init=False, repr=False)  # Raw runner output

    def __attrs_post_init__(self):
        # Initialise runner
        self._kernel_dict = self.scene.kernel_dict()
        self._kernel_dict.check()
        self._runner = OneDimRunner(self._kernel_dict)

    @classmethod
    def new(cls, *args, **kwargs):
        """Create a :class:`.RamiSolverApp` instance after preparatory checks.
        All arguments are forwarded to the :class:`.RamiSolverApp`
        constructor.
        """
        if isinstance(eradiate.mode(), ModeNone):
            raise ModeError(f"no mode selected, use eradiate.set_mode()")

        return cls(*args, **kwargs)

    @classmethod
    def from_dict(cls, d):
        """Instantiate from a dictionary."""
        # Collect mode configuration
        solver_config = pinttr.interpret_units(d, ureg=ureg)

        try:
            mode_config = solver_config.pop("mode")
        except KeyError:
            raise ValueError("section 'mode' missing from configuration "
                             "dictionary")

        try:
            mode_id = mode_config.pop("type")
        except KeyError or AttributeError:
            raise ValueError("parameter 'mode.type' missing from configuration "
                             "dictionary")

        # Select appropriate operational mode
        eradiate.set_mode(mode_id, **mode_config)
        if not eradiate.mode.is_monochromatic():
            raise ModeError("only monochromatic modes are supported")

        # Create scene
        scene = RamiScene(**solver_config)

        # Instantiate class
        return cls.new(scene=scene)

    def process(self):
        """Run simulation on the configured scene. Raw results yielded by the
        encapsulated runner are stored in ``self._raw_results``.

        .. seealso:: :meth:`postprocess`, :meth:`run`
        """
        self._raw_results = None  # Unset raw results for error detection
        self._raw_results = self._runner.run()

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
        sza = ensure_array(illumination.zenith.to(ureg.deg).magnitude, dtype=float)
        cos_sza = np.cos(illumination.zenith.to(ureg.rad).magnitude)
        saa = ensure_array(illumination.azimuth.to(ureg.deg).magnitude, dtype=float)
        wavelength = ensure_array(eradiate.mode().wavelength.magnitude, dtype=float)

        # Format results
        sensor_query = Query()
        for measure in scene.measures:
            # Collect results from sensors associated to processed measure
            measure_id = measure.id
            entries = scene.measure_registry.search(sensor_query.measure_id == measure_id)
            sensor_ids = [db_entry["sensor_id"] for db_entry in entries]
            sensor_spps = [db_entry["sensor_spp"] for db_entry in entries]
            data = measure.postprocess_results(sensor_ids, sensor_spps, self._raw_results)

            if len(data.shape) != 2:
                raise ValueError(f"raw result array has incorrect shape "
                                 f"{data.shape}")

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
                        ("wavelength", wavelength)
                    ),
                )

                # Did the sensor sample ray directions in the hemisphere or on a
                # plane?
                plane = (data.shape[2] == 1)

                # Compute viewing angles at pixel centers
                xs = ds.x
                ys = ds.y
                theta = np.full((len(ys), len(xs)), np.nan)
                phi = np.full_like(theta, np.nan)

                if plane:
                    for x in xs:
                        for y in ys:
                            sample = float(x + 0.5) / len(xs)
                            theta[y, x] = 90. - 180. * sample
                            phi[y, x] = measure.orientation.to("deg").m
                else:
                    for x in xs:
                        for y in ys:
                            xy = [float((x + 0.5) / len(xs)),
                                  float((y + 0.5) / len(ys))]
                            d = square_to_uniform_hemisphere(xy)
                            theta[y, x], phi[y, x] = \
                                direction_to_angles(d).to("deg").m

                # Assign angles as non-dimension coords
                ds["vza"] = (("y", "x"), theta, {"units": "deg"})
                ds["vaa"] = (("y", "x"), phi, {"units": "deg"})
                ds = ds.set_coords(("vza", "vaa"))

            else:
                raise ValueError(f"Unsupported measure type {measure.__class__}")

            # Add other variables
            ds["irradiance"] = (
                ("sza", "saa", "wavelength"),
                np.array(
                    self._kernel_dict["illumination"]["irradiance"]["value"] *
                    cos_sza
                ).reshape((1, 1, 1))
            )
            ds["brdf"] = ds["lo"] / ds["irradiance"]
            ds["brf"] = ds["brdf"] * np.pi

            # Add missing metadata
            dataset_spec = DatasetSpec(
                convention="CF-1.8",
                title="Top-of-canopy simulation results",
                history=f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
                        f"data creation - {__name__}.{self.__class__}.run",
                source=f"eradiate, version {eradiate.__version__}",
                references="",
                var_specs={
                    "irradiance": VarSpec(
                        standard_name="toc_horizontal_solar_irradiance_per_unit_wavelength",
                        units=str(uck.get("irradiance")),
                        long_name="top-of-canopy horizontal spectral irradiance"
                    ),
                    "lo": VarSpec(
                        standard_name="toc_outgoing_radiance_per_unit_wavelength",
                        units=str(uck.get("radiance")),
                        long_name="top-of-canopy outgoing spectral radiance"
                    ),
                    "brf": VarSpec(
                        standard_name="toc_brf",
                        units="dimensionless",
                        long_name="top-of-canopy bi-directional reflectance factor"
                    ),
                    "brdf": VarSpec(
                        standard_name="toc_brdf",
                        units="1/sr",
                        long_name="top-of-canopy bi-directional reflection distribution function"
                    )
                },
                coord_specs="angular_observation"
            )
            ds.ert.normalize_metadata(dataset_spec)

            self.results[measure_id] = ds

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
        for measure_id, results in self.results.items():
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
            result = self.results[measure_id]

            if isinstance(measure, DistantMeasure):
                for quantity in result.data_vars:
                    if quantity == "irradiance":
                        continue

                    data = result[quantity]

                    if data.shape[2] == 1:  # Only a plane is covered
                        data.squeeze().plot(x="vza")
                    else:
                        data.squeeze().plot.pcolormesh()

                    fname_plot = os.path.abspath(f"{fname_prefix}_{measure_id}_{quantity}.png")
                    os.makedirs(os.path.dirname(fname_plot), exist_ok=True)
                    print(f"Saving plot to {fname_plot}")
                    plt.savefig(fname_plot, bbox_inches="tight")
                    plt.close()

            else:
                raise NotImplementedError
