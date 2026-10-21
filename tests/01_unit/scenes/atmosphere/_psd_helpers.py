"""Helpers building :class:`.ParticleStateComponent` and
:class:`.ParticlePSDComponent` test instances."""

import numpy as np
import xarray as xr

from eradiate.data.convert import make_prt_psd_v1, make_prt_v1
from eradiate.grid import CartesianGrid
from eradiate.scenes.atmosphere import ParticlePSDComponent
from eradiate.scenes.geometry import PlaneParallelGeometry
from eradiate.units import unit_registry as ureg

PHAMAT_LABELS = {
    4: ["11", "12", "33", "34"],
    6: ["11", "12", "22", "33", "34", "44"],
}


def make_properties_dataset(
    reff=10.0,
    veff=0.1,
    w=(0.5, 1.5),
    n_theta=5,
    nphamat=4,
    phase_value=1.0,
    ext_value=1.0,
    ssa_value=0.8,
) -> xr.Dataset:
    """
    Build a Prt v1 properties dataset (with ``reff``/``veff`` dimensions)
    directly, with a shared ascending mu grid across the whole
    ``(w, reff, veff)`` grid.
    """
    reff_values = np.atleast_1d(np.asarray(reff, dtype=float))
    veff_values = np.atleast_1d(np.asarray(veff, dtype=float))
    w_values = np.atleast_1d(np.asarray(w, dtype=float))
    nw, nreff, nveff = len(w_values), len(reff_values), len(veff_values)

    mu_1d = np.linspace(-1.0, 1.0, n_theta)
    mu = np.tile(mu_1d, (nw, nreff, nveff, 1))
    theta = np.degrees(np.arccos(mu))

    ext = np.broadcast_to(np.asarray(ext_value, dtype=float), (nw, nreff, nveff)).copy()
    ssa = np.broadcast_to(np.asarray(ssa_value, dtype=float), (nw, nreff, nveff)).copy()
    phase = np.broadcast_to(
        np.asarray(phase_value, dtype=float), (nphamat, nw, nreff, nveff, n_theta)
    ).copy()

    return make_prt_psd_v1(
        w=w_values * ureg.micron,
        phamat=PHAMAT_LABELS[nphamat],
        mu=mu * ureg.dimensionless,
        theta=theta * ureg.degree,
        ext=ext * ureg("km^-1 / (g / m^3)"),
        ssa=ssa * ureg.dimensionless,
        phase=phase * ureg("1/sr"),
        reff=reff_values * ureg.micron,
        veff=veff_values * ureg.dimensionless,
    )


def make_state_properties_dataset(
    state_dims: dict,
    w=(1.0,),
    n_theta=5,
    nphamat=4,
    ssa_value=0.8,
) -> xr.Dataset:
    """
    Build a Prt-Core v1 properties dataset extended with arbitrary state
    dimensions, with a shared ascending mu grid and an isotropic phase
    function.

    The per-unit-amount extinction, in km⁻¹ / (g/m³), is
    ``1 + sum(state coordinate magnitudes)``: linear along every state
    dimension, so linear interpolation reproduces it exactly.

    Parameters
    ----------
    state_dims : dict
        Mapping of state dimension names to their coordinate values
        (quantities), forwarded to :func:`.make_prt_v1`.
    """
    w_values = np.atleast_1d(np.asarray(w, dtype=float))
    nw = len(w_values)
    coords = [v.m for v in state_dims.values()]
    shape = tuple(len(c) for c in coords)

    ext = 1.0 + sum(np.meshgrid(*coords, indexing="ij"))
    ext = np.broadcast_to(ext, (nw, *shape)).copy()
    ssa = np.full((nw, *shape), ssa_value)
    mu = np.broadcast_to(np.linspace(-1.0, 1.0, n_theta), (nw, *shape, n_theta))
    phase = np.ones((nphamat, nw, *shape, n_theta))

    return make_prt_v1(
        w=w_values * ureg.micron,
        phamat=PHAMAT_LABELS[nphamat],
        mu=mu * ureg.dimensionless,
        theta=np.degrees(np.arccos(mu)) * ureg.degree,
        ext=ext * ureg("km^-1 / (g / m^3)"),
        ssa=ssa * ureg.dimensionless,
        phase=phase * ureg("1/sr"),
        state_dims=state_dims,
    )


def make_state_profile_dataset(grid: CartesianGrid, variables: dict) -> xr.Dataset:
    """
    Build a dense, spatially uniform Pfd-Core v1 profile dataset filling every
    voxel of ``grid``.

    Parameters
    ----------
    grid : CartesianGrid
        Grid whose cells the profile fills.

    variables : dict
        Mapping of profile variable names to their (scalar) values
        (quantities).
    """
    shape = (grid.n_cells_z, grid.n_cells_y, grid.n_cells_x)
    return xr.Dataset(
        data_vars={
            name: (["z", "y", "x"], np.full(shape, q.m), {"units": str(q.u)})
            for name, q in variables.items()
        },
        coords={
            "x_edges": (["x_edges"], grid.edges_x.m_as("km"), {"units": "kilometer"}),
            "y_edges": (["y_edges"], grid.edges_y.m_as("km"), {"units": "kilometer"}),
            "z_edges": (["z_edges"], grid.levels.m_as("km"), {"units": "kilometer"}),
        },
    )


def make_profile_dataset(
    grid: CartesianGrid,
    reff=10.0,
    veff=0.1,
    mass_concentration=1e-3,
    z_edges=None,
) -> xr.Dataset:
    """Build a dense Pfd-Core v1 profile dataset filling every voxel of ``grid``."""
    n_x, n_y = grid.n_cells_x, grid.n_cells_y
    z_edges_km = (
        np.asarray(z_edges, dtype=float)
        if z_edges is not None
        else grid.levels.m_as("km")
    )
    n_z = len(z_edges_km) - 1
    shape = (n_z, n_y, n_x)

    def _to_zyx(value):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1 and arr.shape[0] == n_z:
            arr = arr.reshape(n_z, 1, 1)
        return np.broadcast_to(arr, shape).copy()

    data_vars = {
        "reff": (["z", "y", "x"], _to_zyx(reff), {"units": "micron"}),
        "veff": (["z", "y", "x"], _to_zyx(veff), {"units": "dimensionless"}),
        "mass_concentration": (
            ["z", "y", "x"],
            _to_zyx(mass_concentration),
            {"units": "g/m^3"},
        ),
    }
    coords = {
        "x_edges": (["x_edges"], grid.edges_x.m_as("km"), {"units": "kilometer"}),
        "y_edges": (["y_edges"], grid.edges_y.m_as("km"), {"units": "kilometer"}),
        "z_edges": (["z_edges"], z_edges_km, {"units": "kilometer"}),
    }

    return xr.Dataset(data_vars=data_vars, coords=coords)


def make_particle_psd_component(
    n_x: int = 2,
    n_y: int = 2,
    n_z: int = 3,
    x_extent: ureg.Quantity = 2.0 * ureg.km,
    y_extent: ureg.Quantity = 2.0 * ureg.km,
    z_extent: ureg.Quantity = 3.0 * ureg.km,
    reff=10.0,
    veff=0.1,
    mass_concentration=1e-3,
    w=(0.5, 1.5),
    ext_value=1.0,
    ssa_value=0.8,
    has_absorption=True,
    has_scattering=True,
) -> ParticlePSDComponent:
    """Build a dense, spatially-uniform :class:`.ParticlePSDComponent`."""
    grid = CartesianGrid(
        edges_x=np.linspace(0.0, x_extent.m_as("km"), n_x + 1) * ureg.km,
        edges_y=np.linspace(0.0, y_extent.m_as("km"), n_y + 1) * ureg.km,
        levels=np.linspace(0.0, z_extent.m_as("km"), n_z + 1) * ureg.km,
    ).centered()
    geometry = PlaneParallelGeometry(grid=grid, toa_altitude=grid.levels[-1])

    return ParticlePSDComponent(
        profile=make_profile_dataset(
            grid, reff=reff, veff=veff, mass_concentration=mass_concentration
        ),
        properties=make_properties_dataset(
            reff=reff, veff=veff, w=w, ext_value=ext_value, ssa_value=ssa_value
        ),
        geometry=geometry,
        has_absorption=has_absorption,
        has_scattering=has_scattering,
    )


def make_z_profile_particle_psd_component(z_edges_src, target_levels):
    """Build a single-reff/veff :class:`.ParticlePSDComponent` with a z-varying
    profile, resampled onto ``target_levels``."""
    z_edges_src = np.asarray(z_edges_src, dtype=float)
    z_centers_src = 0.5 * (z_edges_src[:-1] + z_edges_src[1:])

    target_grid = CartesianGrid(
        edges_x=np.array([-1.0, 1.0]) * ureg.km,
        edges_y=np.array([-1.0, 1.0]) * ureg.km,
        levels=np.asarray(target_levels, dtype=float) * ureg.km,
    )
    geometry = PlaneParallelGeometry(
        grid=target_grid,
        toa_altitude=target_grid.levels[-1],
        ground_altitude=target_grid.levels[0],
    )
    profile = make_profile_dataset(
        target_grid,
        reff=10.0 + z_centers_src,
        veff=0.1,
        mass_concentration=z_centers_src,
        z_edges=z_edges_src,
    )
    # Properties has a single reff/veff point, so ext/ssa interpolation is
    # constant regardless of the profile's own per-voxel reff, which is
    # therefore free to vary with z here.
    psd = ParticlePSDComponent(
        profile=profile, properties=make_properties_dataset(), geometry=geometry
    )
    return psd, target_grid
