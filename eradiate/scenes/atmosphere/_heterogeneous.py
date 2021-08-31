from __future__ import annotations

import tempfile
from pathlib import Path
from typing import MutableMapping, Optional

import attr
import numpy as np
import pint

from ._core import (
    Atmosphere,
    atmosphere_factory,
    read_binary_grid3d,
    write_binary_grid3d,
)
from ...attrs import AUTO, documented, parse_docs
from ...contexts import KernelDictContext
from ...kernel.transform import map_cube, map_unit_cube
from ...radprops import rad_profile_factory
from ...radprops.rad_profile import RadProfile, US76ApproxRadProfile
from ...units import to_quantity
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


@atmosphere_factory.register(type_id="heterogeneous")
@parse_docs
@attr.s
class HeterogeneousAtmosphere(Atmosphere):
    """
    Heterogeneous atmosphere scene element [``heterogeneous``].

    This class builds a one-dimensional heterogeneous atmosphere.
    It expands as a ``heterogeneous`` kernel plugin, which takes as parameters
    a phase function and a set of paths to volume data files.
    The radiative properties used to configure
    :class:`.HeterogeneousAtmosphere` are specified by a :class:`.RadProfile`
    object.
    The vertical extension of the atmosphere is automatically adjusted to
    match that of the :class:`.RadProfile` object.
    The atmosphere's bottom altitude is set to 0 km.
    The phase function is set to :class:`.RayleighPhaseFunction`.
    """

    profile: RadProfile = documented(
        attr.ib(
            default=attr.Factory(US76ApproxRadProfile),
            converter=rad_profile_factory.convert,
            validator=attr.validators.instance_of(RadProfile),
        ),
        doc="Radiative property profile used. If set, volume data files will be "
        "created from profile data to initialise the corresponding kernel "
        "plugin.",
        type=":class:`~eradiate.radprops.rad_profile.RadProfile`",
        default=":class:`US76ApproxRadProfile() <.US76ApproxRadProfile>`",
    )

    albedo_filename: str = documented(
        attr.ib(
            default="albedo.vol",
            converter=str,
            validator=attr.validators.instance_of(str),
        ),
        doc="Name of the albedo volume data file.",
        type="str",
        default='"albedo.vol"',
    )

    sigma_t_filename: str = documented(
        attr.ib(
            default="sigma_t.vol",
            converter=str,
            validator=attr.validators.instance_of(str),
        ),
        doc="Name of the extinction coefficient volume data file.",
        type="str",
        default='"sigma_t.vol"',
    )

    cache_dir: Path = documented(
        attr.ib(
            default=Path(tempfile.mkdtemp()),
            converter=Path,
            validator=attr.validators.instance_of(Path),
        ),
        doc="Path to a cache directory where volume data files will be created.",
        type="path-like",
        default="Temporary directory",
    )

    _quantities = {"albedo": "albedo", "sigma_t": "collision_coefficient"}

    def __attrs_post_init__(self):
        # Prepare cache directory in case we'd need it
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    #                             Properties
    # --------------------------------------------------------------------------

    @property
    def albedo_file(self) -> Path:
        return self.cache_dir / self.albedo_filename

    @property
    def sigma_t_file(self) -> Path:
        return self.cache_dir / self.sigma_t_filename

    @property
    def bottom(self) -> pint.Quantity:
        return ureg.Quantity(0.0, "km")

    @property
    def top(self) -> pint.Quantity:
        return self.profile.levels.max()

    # --------------------------------------------------------------------------
    #                       Evaluation methods
    # --------------------------------------------------------------------------

    def eval_width(self, ctx: KernelDictContext) -> pint.Quantity:
        if self.width is AUTO:
            spectral_ctx = ctx.spectral_ctx if ctx is not None else None

            if self.profile is None:
                albedo = ureg.Quantity(
                    read_binary_grid3d(self.albedo_filename),
                    ureg.dimensionless,
                )
                sigma_t = ureg.Quantity(
                    read_binary_grid3d(self.sigma_t_filename),
                    uck.get("collision_coefficient"),
                )
                min_sigma_s = (sigma_t * albedo).min()
            else:
                min_sigma_s = self.profile.eval_sigma_s(spectral_ctx).min()

            if min_sigma_s <= 0.0:
                raise ValueError(
                    "cannot compute width automatically when scattering "
                    "coefficient reaches zero"
                )

            return min(10.0 / min_sigma_s, ureg.Quantity(1e3, "km"))

        else:
            return self.width

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    def kernel_phase(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
        return {f"phase_{self.id}": {"type": "rayleigh"}}

    def kernel_media(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:

        length_units = uck.get("length")
        width = self.kernel_width(ctx).m_as(length_units)
        top = self.top.m_as(length_units)
        bottom = self.bottom.m_as(length_units)
        trafo = map_unit_cube(
            xmin=-width / 2.0,
            xmax=width / 2.0,
            ymin=-width / 2.0,
            ymax=width / 2.0,
            zmin=bottom,
            zmax=top,
        )

        radprops = self.profile.to_dataset(spectral_ctx=ctx.spectral_ctx)
        albedo = to_quantity(radprops.albedo).m_as(uck.get("albedo"))
        sigma_t = to_quantity(radprops.sigma_t).m_as(uck.get("collision_coefficient"))
        write_binary_grid3d(
            filename=str(self.albedo_file), values=albedo[np.newaxis, np.newaxis, ...]
        )
        write_binary_grid3d(
            filename=str(self.sigma_t_file), values=sigma_t[np.newaxis, np.newaxis, ...]
        )

        return {
            f"medium_{self.id}": {
                "type": "heterogeneous",
                "phase": {"type": "rayleigh"},
                "sigma_t": {
                    "type": "gridvolume",
                    "filename": str(self.sigma_t_file),
                    "to_world": trafo,
                },
                "albedo": {
                    "type": "gridvolume",
                    "filename": str(self.albedo_file),
                    "to_world": trafo,
                },
            }
        }

    def kernel_shapes(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
        if ctx.ref:
            medium = {"type": "ref", "id": f"medium_{self.id}"}
        else:
            medium = self.kernel_media(ctx=None)[f"medium_{self.id}"]

        length_units = uck.get("length")
        width = self.kernel_width(ctx).m_as(length_units)
        bottom = self.bottom.m_as(length_units)
        top = self.top.m_as(length_units)
        offset = self.kernel_offset(ctx).m_as(length_units)
        trafo = map_cube(
            xmin=-width / 2.0,
            xmax=width / 2.0,
            ymin=-width / 2.0,
            ymax=width / 2.0,
            zmin=bottom - offset,
            zmax=top,
        )

        return {
            f"shape_{self.id}": {
                "type": "cube",
                "to_world": trafo,
                "bsdf": {"type": "null"},
                "interior": medium,
            }
        }
