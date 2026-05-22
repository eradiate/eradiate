from __future__ import annotations

import attrs

import eradiate

from ._core import Integrator
from ...attrs import define, documented


@define(eq=False, slots=False)
class MonteCarloIntegrator(Integrator):
    """
    Base class for integrator elements wrapping kernel classes
    deriving from
    :class:`mitsuba.MonteCarloIntegrator`.

    .. warning:: This class should not be instantiated.
    """

    max_depth: int | None = documented(
        attrs.field(default=None, converter=attrs.converters.optional(int)),
        doc="Longest path depth in the generated measure data (where -1 "
        "corresponds to ∞). A value of 1 will display only visible emitters. 2 "
        "computes only direct illumination (no multiple scattering), etc. If "
        "unset, the kernel default value (-1) is used.",
        type="int or None",
        init_type="int, optional",
    )

    rr_depth: int = documented(
        attrs.field(default=5, converter=int),
        doc="Minimum path depth after which the implementation starts applying "
        "the Russian roulette path termination criterion.",
        type="int",
    )

    hide_emitters: bool | None = documented(
        attrs.field(default=None, converter=attrs.converters.optional(bool)),
        doc="Hide directly visible emitters. If unset, the kernel default "
        "value (``false``) is used.",
        type="bool or None",
        init_type="bool, optional",
    )

    @property
    def extremum_compatible(self):
        """
        Returns
        -------
        bool
            Flags whether the integrator can consume extremum structures.
        """
        return False

    @property
    def kernel_type(self) -> str:
        raise NotImplementedError

    def _build_kernel_dict(self) -> dict:
        """
        Build the kernel-specific dictionary.

        Override this method in subclasses to add integrator-specific parameters.
        The base implementation handles common Monte Carlo integrator parameters.
        """
        result = {
            "type": self.kernel_type,
            "rr_depth": self.rr_depth,
        }

        if self.timeout is not None:
            result["timeout"] = self.timeout
        if self.max_depth is not None:
            result["max_depth"] = self.max_depth
        if self.hide_emitters is not None:
            result["hide_emitters"] = self.hide_emitters

        return result

    @property
    def template(self) -> dict:
        # Validation
        if self.stokes and not eradiate.mode().is_polarized:
            raise RuntimeError("stokes should only be set to True in polarized mode.")

        # Build the kernel dict (children can override _build_kernel_dict)
        result = self._build_kernel_dict()

        if self.stokes or self.moment:
            result = {
                "type": "stokes_moment",
                "nested": result,
                "use_stokes": self.stokes,
                "use_moment": self.moment,
                "meridian_align": self.meridian_align,
            }

        return result


@define(eq=False, slots=False)
class PathIntegrator(MonteCarloIntegrator):
    """
    A thin interface to the path tracer kernel plugin [``path``].

    This integrator samples paths using random walks starting from the sensor.
    It supports multiple scattering and does not account for volume
    interactions.
    """

    @property
    def kernel_type(self) -> str:
        return "path"


@define(eq=False, slots=False)
class VolPathIntegrator(MonteCarloIntegrator):
    """
    A thin interface to the volumetric path tracer kernel plugin [``volpath``].

    This integrator samples paths using random walks starting from the sensor.
    It supports multiple scattering and accounts for volume interactions.
    """

    @property
    def kernel_type(self) -> str:
        return "volpath"


@define(eq=False, slots=False)
class EOVolPathIntegrator(MonteCarloIntegrator):
    """
    A thin interface to the EO volumetric path tracer kernel plugin [``eovolpath``].

    This integrator samples paths using random walks starting from the sensor.
    It supports multiple scattering and accounts for volume interactions. It
    implements all the variance reduction methods from VROOM
    :cite:t:`Buras2011EfficientUnbiasedVariance`: DDIS, prediction-based path
    splitting (PBS), and the Non-Local Estimator (NLE). The default values
    are set to the ones suggested in the article to the exception of
    ``pbs_max_split_count``, which has been decreased to avoid drastic slowdowns
    and ``nle_first_clone_depth`` to accomodate how depth is calculated in the
    kernel.

    It also supports the use of extremum structures and the estimation of
    transmittance through residual ratio tracking :cite:`Novak2014Residual`.
    The former can improve performance in heterogeneous atmospheres, and the
    latter reduces variance in transmittance estimation.
    """

    rr_depth: int = documented(
        attrs.field(
            default=1000,
            converter=int,
            validator=attrs.validators.instance_of(int),
        ),
        doc="Minimum path depth after which the implementation starts applying "
        "the Russian roulette path termination criterion.",
        type="int",
    )

    rr_factor = documented(
        attrs.field(
            default=0.97,
            converter=float,
            validator=attrs.validators.instance_of(float),
        ),
        doc="Maximum probability of keeping a path when Russian Roulette is evaluated.",
        type="float",
    )

    vroom_enable = documented(
        attrs.field(
            default=False,
            converter=bool,
            validator=attrs.validators.instance_of(bool),
        ),
        doc="Activate all VROOM variance reduction methods (DDIS, PBS, and NLE). "
        "Overrides ``ddis_enable``, ``pbs_enable``, and ``nle_enable``.",
    )

    ddis_enable = documented(
        attrs.field(
            default=False,
            converter=bool,
            validator=attrs.validators.instance_of(bool),
        ),
        doc="Activate the DDIS variance reduction method. The ``ddis_threshold`` "
        "controlling the probability of sampling using the emitter direction is "
        "set in the :class:`eradiate.scenes.atmosphere.Atmosphere` interface.",
        type="bool",
    )

    ddis_enable_surface = documented(
        attrs.field(
            default=True,
            converter=bool,
            validator=attrs.validators.instance_of(bool),
        ),
        doc="Apply DDIS to surfaces when ``ddis_enable=True``. Uses the same "
        "``ddis_threshold`` as mentioned in ``ddis_enable``.",
        type="bool",
    )

    pbs_enable = documented(
        attrs.field(
            default=False,
            converter=bool,
            validator=attrs.validators.instance_of(bool),
        ),
        doc="Enable prediction-based path splitting (PBS). At each volumetric "
        "scattering event, the predicted contribution of the scattered direction "
        "is used to decide whether to split the path into multiple independent "
        "copies or to apply Russian roulette. Each split copy carries a "
        "proportionally reduced weight.",
        type="bool",
    )

    pbs_min_split_threshold = documented(
        attrs.field(
            default=3.0,
            converter=float,
            validator=attrs.validators.instance_of(float),
        ),
        doc="Minimum prediction weight required to trigger a split. Only paths "
        "whose predicted weight exceeds this value are split. Must be greater "
        "than 1 for splitting to produce more than one copy.",
        type="float",
    )

    pbs_max_split_count = documented(
        attrs.field(
            default=50,
            converter=int,
            validator=attrs.validators.instance_of(int),
        ),
        doc="Maximum number of path copies created at a single splitting event. "
        "The actual count is ``min(pbs_max_split_count, floor(w_spl))``, where "
        "``w_spl`` is the split prediction weight.",
        type="int",
    )

    pbs_crit_rr_threshold = documented(
        attrs.field(
            default=0.33,
            converter=float,
            validator=attrs.validators.instance_of(float),
        ),
        doc="Prediction weight threshold below which Russian roulette is applied "
        "to split paths. Split paths whose current prediction weight falls below "
        "this value are stochastically terminated to limit the cost of low-weight "
        "copies.",
        type="float",
    )

    pbs_min_rr_threshold = documented(
        attrs.field(
            default=0.2,
            converter=float,
            validator=attrs.validators.instance_of(float),
        ),
        doc="Minimum survival probability for split-path Russian roulette. The "
        "survival probability is ``max(w_spl, pbs_min_rr_threshold)``, ensuring the "
        "kill probability never exceeds ``1 - pbs_min_rr_threshold`` and that "
        "surviving paths are not reweighted above their pre-split weight.",
        type="float",
    )

    nle_enable = documented(
        attrs.field(
            default=False,
            converter=bool,
            validator=attrs.validators.instance_of(bool),
        ),
        doc="Enable the Non-Local Estimator (NLE) variance reduction method. "
        "Each primary ray is traced as a *mother* path. At regular intervals "
        "along the mother's trajectory, a *clone* path is forked and traced "
        "independently to perform additional next-event estimation. The mother's "
        "own NEE contributions are restricted to avoid double-counting.",
        type="bool",
    )

    nle_first_clone_depth = documented(
        attrs.field(
            default=5,
            converter=int,
            validator=attrs.validators.instance_of(int),
        ),
        doc="Scatter depth at which the mother path creates its first clone. "
        "Clone creation then recurs every ``nee_per_clone`` scatters thereafter.",
        type="int",
    )

    nle_max_clone_depth = documented(
        attrs.field(
            default=12,
            converter=int,
            validator=attrs.validators.instance_of(int),
        ),
        doc="Maximum number of scattering events a clone is allowed to trace "
        "before it is terminated. Controls the amount of next-event estimation "
        "work performed per clone.",
        type="int",
    )

    nle_nee_per_clone = documented(
        attrs.field(
            default=11,
            converter=int,
            validator=attrs.validators.instance_of(int),
        ),
        doc="Interval, in scattering events, between successive clone creation "
        "events along the mother path. A new clone is spawned every "
        "``nle_nee_per_clone`` scatters starting from ``nle_first_clone_depth``.",
        type="int",
    )

    @property
    def extremum_compatible(self):
        return True

    @property
    def kernel_type(self) -> str:
        return "eovolpath"

    def _build_kernel_dict(self) -> dict:
        result = super()._build_kernel_dict()
        result["rr_factor"] = self.rr_factor

        result["ddis_enable"] = True if self.vroom_enable else self.ddis_enable
        result["ddis_enable_surface"] = self.ddis_enable_surface

        result["pbs_enable"] = True if self.vroom_enable else self.pbs_enable
        result["pbs_min_split_threshold"] = self.pbs_min_split_threshold
        result["pbs_max_split_count"] = self.pbs_max_split_count
        result["pbs_crit_rr_threshold"] = self.pbs_crit_rr_threshold
        result["pbs_min_rr_threshold"] = self.pbs_min_rr_threshold

        result["nle_enable"] = True if self.vroom_enable else self.nle_enable
        result["nle_first_clone_depth"] = self.nle_first_clone_depth
        result["nle_max_clone_depth"] = self.nle_max_clone_depth
        result["nle_nee_per_clone"] = self.nle_nee_per_clone

        return result


@define(eq=False, slots=False)
class VolPathMISIntegrator(MonteCarloIntegrator):
    """
    A thin interface to the volumetric path tracer kernel plugin [``volpathmis``].

    This plugin implements spectral multiple importance sampling
    :cite:`Miller2019NullscatteringPathIntegral`.
    """

    use_spectral_mis = attrs.field(
        default=None, converter=attrs.converters.optional(bool)
    )

    @property
    def kernel_type(self) -> str:
        return "volpathmis"

    def _build_kernel_dict(self) -> dict:
        result = super()._build_kernel_dict()
        if self.use_spectral_mis is not None:
            result["use_spectral_mis"] = self.use_spectral_mis
        return result


@define(eq=False, slots=False)
class PiecewiseVolPathIntegrator(MonteCarloIntegrator):
    """
    A thin interface to the piecewise volumetric path tracer kernel plugin [``piecewise_volpath``].

    This integrator samples paths using random walks starting from the sensor.
    It supports multiple scattering and accounts for 1D volume interactions.
    """

    @property
    def kernel_type(self) -> str:
        return "piecewise_volpath"
