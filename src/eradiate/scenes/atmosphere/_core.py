from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import ClassVar, Literal

import attrs
import numpy as np
import pint
import xarray as xr
from axsdb import AbsorptionDatabase

from ..geometry import (
    SceneGeometry,
)
from ..phase import PhaseFunction
from ..._factory import Factory
from ...attrs import define, documented
from ...grid import SpatialGrid
from ...spectral.index import SpectralIndex
from ...units import symbol
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg
from ...util.deprecation import deprecated_kwargs

atmosphere_factory = Factory()
atmosphere_factory.register_lazy_batch(
    [
        (
            "_atmosphere.Atmosphere",
            "atmosphere",
            {},
        ),
        (
            "_heterogeneous.HeterogeneousAtmosphere",
            "heterogeneous",
            {},
        ),
        (
            "_homogeneous.HomogeneousComponent",
            "homogeneous",
            {},
        ),
        (
            "_molecular.MolecularComponent",
            "molecular",
            {},
        ),
        (
            "_particle_component.ParticleOTComponent",
            "particle_ot",
            {"aliases": ["particle_layer"]},
        ),
        (
            "_particle_component.ParticleExtinctionComponent",
            "particle_extinction",
            {},
        ),
        (
            "_particle_state_component.ParticleStateComponent",
            "particle_state",
            {},
        ),
        (
            "_particle_state_component.ParticleMSComponent",
            "particle_ms",
            {},
        ),
    ],
    cls_prefix="eradiate.scenes.atmosphere",
)


class AtmosphereComponentFlag(enum.Flag):
    """
    Flags characterizing an atmosphere component.
    """

    NONE = 0
    HOMOGENEOUS = enum.auto()
    MOLECULAR = enum.auto()
    PARTICLE = enum.auto()


class EvalRadProps(ABC):
    """
    Radiative property evaluation interface shared by atmospheres and
    atmosphere components. Implementors must have a ``geometry`` attribute.
    """

    @property
    def absorption_data(self) -> AbsorptionDatabase | None:
        """
        Returns
        -------
        .AbsorptionDatabase or None
            If relevant, the molecular absorption database associated with this
            atmosphere.
        """
        return None

    def eval_radprops(
        self,
        si: SpectralIndex,
        grid: SpatialGrid | None = None,
        optional_fields: bool = False,
    ) -> xr.Dataset:
        """
        Evaluate the extinction coefficients and albedo profiles.

        Parameters
        ----------
        si : .SpectralIndex
            Spectral index.

        grid : .SpatialGrid, optional
            Altitude grid on which evaluation is performed. If unset, an
            instance-specific default is used
            (see :meth:`grid <.Atmosphere.geometry.grid>`).

        optional_fields : bool, optional, default: False
            If ``True``, also output the absorption and scattering coefficients,
            not required for scene setup but useful for analysis and debugging.

        Returns
        -------
        Dataset
            A dataset with the following variables:

            * ``sigma_t``: extinction coefficient;
            * ``albedo``: albedo;
            * ``sigma_a``: absorption coefficient (optional);
            * ``sigma_s``: scattering coefficient (optional).

            with dimensions ``(x_cell, y_cell, z_layer)`` and coordinates:

            * ``x_cell``: X cell centres (azimuth for a spherical grid);
            * ``y_cell``: Y cell centres (colatitude for a spherical grid);
            * ``z_layer``: layer centre altitude.
        """
        if grid is None:
            grid = self.geometry.grid

        sigma_units = ucc.get("collision_coefficient")
        sigma_t = self.eval_sigma_t(si, grid)
        sigma_t = np.broadcast_to(sigma_t, grid.shape)
        albedo = self.eval_albedo(si, grid).m_as(ureg.dimensionless)
        albedo = np.broadcast_to(albedo, grid.shape)

        data_vars = {
            "sigma_t": (
                ("x_cell", "y_cell", "z_layer"),
                sigma_t.m_as(sigma_units),
                {
                    "units": f"{symbol(sigma_units)}",
                    "standard_name": "extinction_coefficient",
                    "long_name": "extinction coefficient",
                },
            ),
            "albedo": (
                ("x_cell", "y_cell", "z_layer"),
                albedo,
                {
                    "standard_name": "albedo",
                    "long_name": "albedo",
                    "units": "",
                },
            ),
        }

        if optional_fields:
            data_vars.update(
                {
                    "sigma_a": (
                        ("x_cell", "y_cell", "z_layer"),
                        sigma_t.m_as(sigma_units) * (1.0 - albedo),
                        {
                            "units": f"{symbol(sigma_units)}",
                            "standard_name": "absorption_coefficient",
                            "long_name": "absorption coefficient",
                        },
                    ),
                    "sigma_s": (
                        ("x_cell", "y_cell", "z_layer"),
                        sigma_t.m_as(sigma_units) * albedo,
                        {
                            "units": f"{symbol(sigma_units)}",
                            "standard_name": "scattering_coefficient",
                            "long_name": "scattering coefficient",
                        },
                    ),
                }
            )

        return xr.Dataset(
            data_vars,
            coords={
                "x_cell": (
                    "x_cell",
                    grid.x_centers.magnitude,
                    {
                        "units": f"{symbol(grid.x_centers.units)}",
                        "standard_name": "cells_x",
                        "long_name": "cells x position",
                    },
                ),
                "y_cell": (
                    "y_cell",
                    grid.y_centers.magnitude,
                    {
                        "units": f"{symbol(grid.y_centers.units)}",
                        "standard_name": "cells_y",
                        "long_name": "cells y position",
                    },
                ),
                "z_layer": (
                    "z_layer",
                    grid.layers.magnitude,
                    {
                        "units": f"{symbol(grid.layers.units)}",
                        "standard_name": "layer_altitude",
                        "long_name": "layer altitude",
                    },
                ),
            },
        )

    @abstractmethod
    def eval_albedo(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        """
        Evaluate albedo spectrum based on a spectral context. This method
        dispatches evaluation to specialized methods depending on the active
        mode.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        grid : .SpatialGrid, optional
            Altitude grid on which evaluation is performed. If unset, an
            instance-specific default is used
            (see :meth:`grid <.Atmosphere.geometry.grid>`).

        Returns
        -------
        quantity
            Evaluated spectrum as an array with length equal to the number of
            layers.
        """

    @abstractmethod
    def eval_sigma_t(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        """
        Evaluate extinction coefficient given a spectral context.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        grid : .SpatialGrid, optional
            Altitude grid on which evaluation is performed. If unset, an
            instance-specific default is used
            (see :meth:`grid <.Atmosphere.geometry.grid>`).

        Returns
        -------
        quantity
            Extinction coefficient.
        """

    @abstractmethod
    def eval_sigma_a(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        """
        Evaluate absorption coefficient given a spectral context.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        grid : .SpatialGrid, optional
            Altitude grid on which evaluation is performed. If unset, an
            instance-specific default is used
            (see :meth:`grid <.Atmosphere.geometry.grid>`).

        Returns
        -------
        quantity
            Absorption coefficient.
        """

    @abstractmethod
    def eval_sigma_s(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        """
        Evaluate scattering coefficient given a spectral context.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        grid : .SpatialGrid, optional
            Altitude grid on which evaluation is performed. If unset, an
            instance-specific default is used
            (see :meth:`grid <.Atmosphere.geometry.grid>`).

        Returns
        -------
        quantity
            Scattering coefficient.
        """

    def eval_transmittance(
        self,
        si: SpectralIndex,
        interaction: Literal["extinction", "absorption", "scattering"] = "extinction",
    ) -> pint.Quantity:
        """
        Evaluate the atmosphere's transmittance with respect to specific
        interaction.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        interaction : {"extinction", "absorption", "scattering"}, optional, default: "extinction"
            Interaction type.

        Returns
        -------
        quantity
            Total atmosphere transmittance.
        """
        eval_sigma = {
            "extinction": self.eval_sigma_t,
            "absorption": self.eval_sigma_a,
            "scattering": self.eval_sigma_s,
        }
        try:
            eval_sigma_func = eval_sigma[interaction]
        except KeyError as e:
            raise ValueError(
                f"invalid interaction type '{interaction}', "
                f"supported: {list(eval_sigma.keys())}"
            ) from e
        sigma = eval_sigma_func(si)
        sigma = np.broadcast_to(sigma.m, self.geometry.grid.shape) * sigma.u
        dz = np.diff(self.geometry.grid.levels)
        tau = np.sum(np.multiply(sigma, dz).to("1"), axis=-1)
        return np.exp(-tau)

    def eval_transmittance_t(self, si: SpectralIndex) -> pint.Quantity:
        return self.eval_transmittance(si=si, interaction="extinction")

    def eval_transmittance_a(self, si: SpectralIndex) -> pint.Quantity:
        return self.eval_transmittance(si=si, interaction="absorption")

    def eval_transmittance_s(self, si: SpectralIndex) -> pint.Quantity:
        return self.eval_transmittance(si=si, interaction="scattering")


@deprecated_kwargs(
    dict.fromkeys(
        [
            "use_rrt",
            "ddis_threshold",
            "rr_threshold",
            "scale",
            "force_majorant",
            "extremum_resolution",
        ]
    ),
    deprecated_in="1.4.0",
    removed_in="2.0.0",
)
@define(eq=False, slots=False)
class AtmosphereComponent(EvalRadProps, ABC):
    """
    Abstract base class for atmosphere components.

    Components provide radiative properties and a phase function. They are
    assembled by an :class:`.Atmosphere`, which generates the corresponding
    kernel scene dictionary. Subclasses describe themselves with the
    :class:`.AtmosphereComponentFlag` set in the ``flags`` class attribute.
    """

    flags: ClassVar[AtmosphereComponentFlag] = AtmosphereComponentFlag.NONE

    id: str | None = documented(
        attrs.field(
            default="atmosphere",
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc="Component identifier. Overridden by the parent :class:`.Atmosphere`.",
        type="str or None",
        init_type="str, optional",
        default='"atmosphere"',
    )

    geometry: SceneGeometry = documented(
        attrs.field(
            default="plane_parallel",
            converter=SceneGeometry.convert,
            validator=attrs.validators.instance_of(SceneGeometry),
        ),
        doc="Scene geometry on which radiative properties are evaluated. "
        "Overridden by the parent :class:`.Atmosphere`.",
        type=".SceneGeometry",
        init_type=".SceneGeometry or dict or str, optional",
        default='"plane_parallel"',
    )

    def __attrs_post_init__(self):
        self.update()

    def update(self) -> None:
        """
        Update internal state.
        """

    @property
    def phase_id(self) -> str:
        """
        Returns
        -------
        str
            ID of the phase function associated with the component.
        """
        return f"phase_{self.id}"

    @property
    @abstractmethod
    def phase(self) -> PhaseFunction:
        """
        Returns
        -------
        .PhaseFunction
            Phase function associated with the component.
        """
