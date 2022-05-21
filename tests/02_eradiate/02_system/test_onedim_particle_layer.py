"""Test cases targeting the particle layer component of a 1D experiment."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import eradiate
import eradiate.scenes as esc
from eradiate import unit_registry as ureg
from eradiate.scenes.atmosphere import ParticleLayer
from eradiate.scenes.phase import TabulatedPhaseFunction
from eradiate.units import symbol, to_quantity


@pytest.fixture
def tabulated_rayleigh():
    """Returns a tabulated version of the Rayleigh phase function."""
    w = np.arange(279.0, 2402, 1) * ureg.nm
    mu = np.linspace(-1.0, 1.0, 201) * ureg.dimensionless

    # Evaluate Rayleigh phase function
    magnitude = lambda mu: 3.0 * (1 + np.square(mu)) / (16 * np.pi)
    eval_rayleigh = (
        np.stack([magnitude(mu) for _ in w], axis=0).reshape(w.size, mu.size, 1, 1)
        / ureg.steradian
    )

    # Format it as an xarray data structure compatible with TabulatedPhaseFunction
    # constructor
    rayleigh_data_array = xr.DataArray(
        eval_rayleigh.magnitude,
        dims=["w", "mu", "i", "j"],
        coords={
            "w": ("w", w.magnitude, {"units": symbol(w.units)}),
            "mu": ("mu", mu.magnitude, {"units": symbol(mu.units)}),
            "i": [0],
            "j": [0],
        },
        attrs={"units": symbol(eval_rayleigh.units)},
    )

    yield TabulatedPhaseFunction(data=rayleigh_data_array)


def make_figure(fname_plot, brf_1, brf_2):
    fig = plt.figure(figsize=(8, 5))
    params = dict(x="vza", ls="dotted", marker=".")
    brf_1.plot(**params)
    brf_2.plot(**params)
    plt.legend(["experiment 1", "experiment 2"])
    plt.title(f"w = {to_quantity(brf_1.w).squeeze():~}")
    plt.tight_layout()
    fig.savefig(fname_plot, dpi=200)
    plt.close()


@pytest.mark.parametrize("w", [280, 400, 550, 650, 1000, 1500, 2400])
@pytest.mark.parametrize("albedo", [0.2, 0.5, 0.8])
@pytest.mark.slow
def test_homogeneous_vs_particle_layer(
    mode_mono_double,
    tmpdir,
    onedim_rayleigh_radprops,
    albedo,
    w,
    artefact_dir,
    ert_seed_state,
):
    r"""
    Equivalency of homogeneous atmosphere and corresponding particle layer
    ======================================================================

    Assert that a homogeneous atmosphere is equivalent to uniform particle layer
    parametrised such that the participating medium has the same radiative
    properties as the homogeneous atmosphere.

    Rationale
    ---------

    * Sensor: Distant measure covering a plane (11 angular points,
      10^6 sample per pixel) and targeting (0, 0, 0).
    * Illumination: Directional illumination with a zenith angle
      :math:`\theta = 30.0°` with black body irradiance spectrum.
    * Surface: a square surface with a Lambertian BRDF with 35 % reflectance.
    * Atmosphere:
      * in experiment 1: a 5 km high homogeneous atmosphere with
        a tabulated phase function corresponding to the Rayleigh phase function.
        The scattering coefficient is given by
        :meth:`eradiate.radprops.rayleigh.compute_sigma_s_air`.
      * in experiment 2: a uniform 5 km thick particle layer with
        a tabulated phase function corresponding to the Rayleigh phase function.
        The optical thickness at 550 nm of the particle layer is set to a value
        that match the optical thickness at 550 nm of the homogeneous
        atmosphere in experiment 1.
    * Integrator: volumetric path tracer.

    Expected behaviour
    ------------------

    Distant bidirectional reflectance factors measured by both experiments must
    agree within 0.5%.

    Results
    -------

    .. image:: generated/plots/test_homogeneous_vs_particle_layer_280.0_0.2.png
       :width: 95%
    .. image:: generated/plots/test_homogeneous_vs_particle_layer_280.0_0.5.png
       :width: 95%
    .. image:: generated/plots/test_homogeneous_vs_particle_layer_280.0_0.8.png
       :width: 95%
    .. image:: generated/plots/test_homogeneous_vs_particle_layer_400.0_0.2.png
       :width: 95%
    .. image:: generated/plots/test_homogeneous_vs_particle_layer_400.0_0.5.png
       :width: 95%
    .. image:: generated/plots/test_homogeneous_vs_particle_layer_400.0_0.8.png
       :width: 95%
    .. image:: generated/plots/test_homogeneous_vs_particle_layer_550.0_0.2.png
       :width: 95%
    .. image:: generated/plots/test_homogeneous_vs_particle_layer_550.0_0.5.png
       :width: 95%
    .. image:: generated/plots/test_homogeneous_vs_particle_layer_550.0_0.8.png
       :width: 95%
    .. image:: generated/plots/test_homogeneous_vs_particle_layer_650.0_0.2.png
       :width: 95%
    .. image:: generated/plots/test_homogeneous_vs_particle_layer_650.0_0.5.png
       :width: 95%
    .. image:: generated/plots/test_homogeneous_vs_particle_layer_650.0_0.8.png
       :width: 95%
    .. image:: generated/plots/test_homogeneous_vs_particle_layer_1000.0_0.2.png
       :width: 95%
    .. image:: generated/plots/test_homogeneous_vs_particle_layer_1000.0_0.5.png
       :width: 95%
    .. image:: generated/plots/test_homogeneous_vs_particle_layer_1000.0_0.8.png
       :width: 95%
    .. image:: generated/plots/test_homogeneous_vs_particle_layer_1500.0_0.2.png
       :width: 95%
    .. image:: generated/plots/test_homogeneous_vs_particle_layer_1500.0_0.5.png
       :width: 95%
    .. image:: generated/plots/test_homogeneous_vs_particle_layer_1500.0_0.8.png
       :width: 95%
    .. image:: generated/plots/test_homogeneous_vs_particle_layer_2400.0_0.2.png
       :width: 95%
    .. image:: generated/plots/test_homogeneous_vs_particle_layer_2400.0_0.5.png
       :width: 95%
    .. image:: generated/plots/test_homogeneous_vs_particle_layer_2400.0_0.8.png
       :width: 95%
    """

    def init_experiment_particle_layer(
        bottom, top, dataset_path, w_ref, tau_ref, r, w, spp
    ):
        return eradiate.experiments.OneDimExperiment(
            measures=esc.measure.MultiDistantMeasure.from_viewing_angles(
                spectral_cfg=esc.measure.MeasureSpectralConfig.new(
                    wavelengths=w,
                ),
                zeniths=np.linspace(-75, 75, 11) * ureg.deg,
                azimuths=0.0 * ureg.deg,
                spp=spp,
            ),
            illumination=esc.illumination.DirectionalIllumination(
                zenith=30 * ureg.deg,
                azimuth=0.0 * ureg.deg,
                irradiance=esc.spectra.SolarIrradianceSpectrum(dataset="blackbody_sun"),
            ),
            atmosphere=esc.atmosphere.HeterogeneousAtmosphere(
                molecular_atmosphere=None,
                particle_layers=esc.atmosphere.ParticleLayer(
                    bottom=bottom,
                    top=top,
                    dataset=dataset_path,
                    w_ref=w_ref,
                    tau_ref=tau_ref,
                ),
            ),
            surface=esc.bsdfs.LambertianBSDF(reflectance=r),
        )

    def init_experiment_homogeneous_atmosphere(
        bottom, top, sigma_a, sigma_s, phase, r, w, spp
    ):
        return eradiate.experiments.OneDimExperiment(
            measures=esc.measure.MultiDistantMeasure.from_viewing_angles(
                spectral_cfg=esc.measure.MeasureSpectralConfig.new(
                    wavelengths=w,
                ),
                zeniths=np.linspace(-75, 75, 11) * ureg.deg,
                azimuths=0.0 * ureg.deg,
                spp=spp,
            ),
            illumination=esc.illumination.DirectionalIllumination(
                zenith=30 * ureg.deg,
                azimuth=0.0 * ureg.deg,
                irradiance=esc.spectra.SolarIrradianceSpectrum(dataset="blackbody_sun"),
            ),
            atmosphere=esc.atmosphere.HomogeneousAtmosphere(
                bottom=bottom,
                top=top,
                sigma_a=sigma_a,
                sigma_s=sigma_s,
                phase=phase,
            ),
            surface=esc.bsdfs.LambertianBSDF(reflectance=r),
        )

    w = w * ureg.nm
    spp = 1e6
    reflectance = 1.0
    bottom = 0.0 * ureg.km
    top = 5.0 * ureg.km

    radprops = onedim_rayleigh_radprops(albedo=albedo)
    w_units = radprops.w.attrs["units"]
    sigma_t = to_quantity(radprops.sigma_t.interp(w=w.m_as(w_units)))
    albedo = to_quantity(radprops.albedo.interp(w=w.m_as(w_units)))
    sigma_s = sigma_t * albedo
    sigma_a = sigma_t - sigma_s
    phase = esc.phase.TabulatedPhaseFunction(data=radprops.phase)
    w_ref = 550.0 * ureg.nm
    sigma_t_ref = to_quantity(
        radprops.sigma_t.interp(
            w=w_ref.m_as(w_units),
        )
    )
    height = top - bottom
    tau_ref = sigma_t_ref * height  # the layer is uniform
    dataset_path = tmpdir / "radprops.nc"
    radprops.to_netcdf(dataset_path)

    experiment_1 = init_experiment_particle_layer(
        bottom=bottom,
        top=top,
        dataset_path=dataset_path,
        w_ref=w_ref,
        tau_ref=tau_ref,
        r=reflectance,
        w=w,
        spp=spp,
    )

    experiment_2 = init_experiment_homogeneous_atmosphere(
        bottom=bottom,
        top=top,
        sigma_a=sigma_a,
        sigma_s=sigma_s,
        phase=phase,
        r=reflectance,
        w=w,
        spp=spp,
    )

    ert_seed_state.reset()
    brf_1 = eradiate.run(experiment_1, seed_state=ert_seed_state)["brf"]

    ert_seed_state.reset()
    brf_2 = eradiate.run(experiment_2, seed_state=ert_seed_state)["brf"]

    # Make figure
    filename = f"test_homogeneous_vs_particle_layer_{w.magnitude}_{albedo.m}.png"
    outdir = os.path.join(artefact_dir, "plots")
    os.makedirs(outdir, exist_ok=True)
    fname_plot = os.path.join(outdir, filename)
    make_figure(fname_plot=fname_plot, brf_1=brf_1, brf_2=brf_2)

    outcome = np.allclose(brf_1.values, brf_2.values, rtol=5e-3)

    if outcome is False:
        print(f"Test failed, see artefact {fname_plot}")
        assert False


@pytest.mark.slow
def test_particle_layer_energy_conservation(
    mode_mono_double, tmpdir, onedim_rayleigh_radprops, artefact_dir, ert_seed_state
):
    r"""
    Energy  conservation in a non-absorbing particle layer
    ======================================================

    We assert that the reflected energy is equivalent to the irradiated energy in
    a scene with non-absorbing atmosphere and surface reflectivity of 1.

    Rationale
    ---------

    * Sensor: Distant flux measure covering the hemisphere (normal is (0, 0, 1),
      1e3 samples per pixel).
    * Illumination: Directional illumination with a zenith angle
      :math:`\theta = 30.0°` with black body irradiance spectrum.
    * Surface: a square surface with a Lambertian BRDF with 100 % reflectance.
    * Atmosphere:
      a non-absorbing uniform 5 km thick particle layer with
      a tabulated phase function corresponding to the Rayleigh phase function.
    * Integrator: volumetric path tracer.

    Expected behaviour
    ------------------

    The retrieved albedo must equal to 1.
    """
    spp = 1e3
    reflectance = 1.0
    bottom = 0.0 * ureg.km
    top = 5.0 * ureg.km

    radprops = onedim_rayleigh_radprops(albedo=1.0)

    w_units = radprops.w.attrs["units"]
    w_ref = 550.0 * ureg.nm
    sigma_t_ref = to_quantity(radprops.sigma_t.interp(w=w_ref.m_as(w_units)))
    height = top - bottom
    tau_ref = sigma_t_ref * height  # the layer is uniform
    dataset_path = tmpdir / "radprops.nc"
    radprops.to_netcdf(dataset_path)

    experiment = eradiate.experiments.OneDimExperiment(
        measures=esc.measure.DistantFluxMeasure(
            film_resolution=(32, 32),
            target=esc.measure.TargetRectangle(
                xmin=-20.0 * ureg.km,
                xmax=20.0 * ureg.km,
                ymin=-20.0 * ureg.km,
                ymax=20.0 * ureg.km,
                z=5.0 * ureg.km,
            ),
            spp=spp,
        ),
        illumination=esc.illumination.DirectionalIllumination(
            zenith=30 * ureg.deg,
            azimuth=0.0 * ureg.deg,
            irradiance=esc.spectra.SolarIrradianceSpectrum(dataset="blackbody_sun"),
        ),
        atmosphere=esc.atmosphere.HeterogeneousAtmosphere(
            molecular_atmosphere=None,
            particle_layers=esc.atmosphere.ParticleLayer(
                bottom=bottom,
                top=top,
                dataset=dataset_path,
                w_ref=w_ref,
                tau_ref=tau_ref,
            ),
        ),
        surface=esc.bsdfs.LambertianBSDF(reflectance=reflectance),
    )

    ert_seed_state.reset()
    albedo = eradiate.run(experiment, seed_state=ert_seed_state)["albedo"]
    assert np.allclose(albedo.values, 1.0, rtol=5e-3)


@pytest.mark.parametrize("w", [400, 550, 1000] * ureg.nm)
@pytest.mark.slow
def test_one_layer_molecular_atmosphere_vs_particle_layer(
    mode_mono_double, tabulated_rayleigh, artefact_dir, ert_seed_state, w
):
    r"""
    Equivalency of one-layer molecular atmosphere and corresponding particle layer
    ==============================================================================

    Assert that a one-layer molecular atmosphere is equivalent to uniform
    particle layer parametrised such that the participating medium has the
    same radiative properties as the one-layer molecular atmosphere.

    Rationale
    ---------

    * Sensor: Distant measure covering a plane (11 angular points,
      10^5 sample per pixel) and targeting (0, 0, 0).
    * Illumination: Directional illumination with a zenith angle
      :math:`\theta = 30.0°` with black body irradiance spectrum.
    * Surface: a square surface with a Lambertian BRDF with 100 % reflectance.
    * Atmosphere:
      * in experiment 1: a 2 km high molecular atmosphere with one layer with
        a tabulated phase function corresponding to the Rayleigh phase function.
        The scattering coefficient is given by
        :meth:`eradiate.radprops.rayleigh.compute_sigma_s_air`.
      * in experiment 2: a uniform 2 km thick particle layer with
        a tabulated phase function corresponding to the Rayleigh phase function.
        The optical thickness at 550 nm of the particle layer is set to a value
        that matches the optical thickness at 550 nm of the molecular
        atmosphere in experiment 1.
    * Integrator: volumetric path tracer.

    Expected behaviour
    ------------------

    Distant bidirectional reflectance factors measured by both experiments must
    agree within 0.5%.

    Results
    -------

    .. image:: generated/plots/test_one_layer_molecular_atmosphere_vs_particle_layer_400.png
       :width: 95%
    .. image:: generated/plots/test_one_layer_molecular_atmosphere_vs_particle_layer_550.png
       :width: 95%
    .. image:: generated/plots/test_one_layer_molecular_atmosphere_vs_particle_layer_1000.png
       :width: 95%
    """

    def molecular_particle_layer(
        exp_mol_atm: eradiate.experiments.OneDimExperiment,
    ) -> ParticleLayer:
        """Creates a particle layer with radiative properties identical to
        a one-layer molecular atmosphere in a 1D experiment, assuming the
        phase function of the latter is a tabulated phase function."""

        # extract molecular atmosphere radiative properties
        spectral_ctx = exp_mol_atm.measures[0].spectral_cfg.spectral_ctxs()[0]
        mol_atm = exp_mol_atm.atmosphere
        mol_atm_radprops = mol_atm.eval_radprops(spectral_ctx)
        mol_atm_albedo = to_quantity(mol_atm_radprops.albedo)

        # compute molecular atmosphere optical thickness
        # step 1/3: extract molecular atmosphere extinction coefficient
        mol_atm_sigma_t = to_quantity(mol_atm_radprops.sigma_t)

        # step 2/3: compute molecular atmosphere thickness
        with xr.set_options(keep_attrs=True):
            mol_atm_dz = to_quantity(mol_atm_radprops.z_level.diff(dim="z_level"))
        # step 3/3: compute molecular atmosphere optical thickness
        mol_atm_tau = (mol_atm_sigma_t * mol_atm_dz).to("dimensionless")

        # create particle radiative properties data set and fill it with the
        # molecular atmosphere radiative properties
        # the wavelength dimension does not play any role here since we set
        # all spectra to uniform spectra
        particle_phase = mol_atm.phase.data
        mu = particle_phase.mu.values
        w = to_quantity(particle_phase.w)
        particle_sigma_t = mol_atm_sigma_t * np.ones_like(w)
        particle_albedo = mol_atm_albedo * np.ones_like(w)

        phase = to_quantity(particle_phase)
        particle_radprops = xr.Dataset(
            data_vars={
                "sigma_t": (
                    "w",
                    particle_sigma_t.m,
                    {
                        "standard_name": "air_volume_extinction_coefficient",
                        "long_name": "extinction coefficient",
                        "units": symbol(particle_sigma_t.units),
                    },
                ),
                "albedo": (
                    "w",
                    particle_albedo.m,
                    {
                        "standard_name": "single_scattering_albedo",
                        "long_name": "albedo",
                        "units": symbol(particle_albedo.units),
                    },
                ),
                "phase": (
                    ("w", "mu", "i", "j"),
                    phase.m,
                    {
                        "standard_name": "scattering_phase_matrix",
                        "long_name": "phase matrix",
                        "units": symbol(phase.units),
                    },
                ),
            },
            coords={
                "w": (
                    "w",
                    w.magnitude,
                    {
                        "standard_name": "radiation_wavelength",
                        "long_name": "wavelength",
                        "units": symbol(w.units),
                    },
                ),
                "mu": (
                    "mu",
                    mu,
                    {
                        "standard_name": "scattering_angle_cosine",
                        "long_name": "scattering angle cosine",
                        "units": "dimensionless",
                    },
                ),
                "i": ("i", [0]),
                "j": ("j", [0]),
            },
        )

        return ParticleLayer(
            bottom=mol_atm.bottom,
            top=mol_atm.top,
            w_ref=550.0 * ureg.nm,
            tau_ref=mol_atm_tau,
            dataset=particle_radprops,
            n_layers=1,
        )

    def init_experiment(spp, r, w, atmosphere):
        return eradiate.experiments.OneDimExperiment(
            measures=esc.measure.MultiDistantMeasure.from_viewing_angles(
                spectral_cfg=esc.measure.MeasureSpectralConfig.new(
                    wavelengths=w,
                ),
                zeniths=np.linspace(-75, 75, 11) * ureg.deg,
                azimuths=0.0 * ureg.deg,
                spp=spp,
            ),
            illumination=esc.illumination.DirectionalIllumination(
                zenith=30 * ureg.deg,
                azimuth=0.0 * ureg.deg,
                irradiance=esc.spectra.SolarIrradianceSpectrum(dataset="blackbody_sun"),
            ),
            atmosphere=atmosphere,
            surface=esc.bsdfs.LambertianBSDF(reflectance=r),
        )

    reflectance = 1.0
    spp = 1e5

    # create the 1D experiment with a one-layer molecular atmosphere
    experiment_1 = init_experiment(
        spp=spp,
        r=reflectance,
        w=w,
        atmosphere=esc.atmosphere.MolecularAtmosphere.ussa_1976(
            levels=[0.0, 2.0] * ureg.km, phase=tabulated_rayleigh
        ),
    )

    # create a particle layer "equivalent" to the molecular atmosphere
    particle_layer = molecular_particle_layer(experiment_1)

    # create the 1D experiemtn with the equivalent particle layer
    experiment_2 = init_experiment(
        spp=spp, r=reflectance, w=w, atmosphere=particle_layer
    )

    # run the experiments
    ert_seed_state.reset()
    brf_1 = eradiate.run(experiment_1, seed_state=ert_seed_state)["brf"]

    ert_seed_state.reset()
    brf_2 = eradiate.run(experiment_2, seed_state=ert_seed_state)["brf"]

    # make figure
    filename = f"test_one_layer_molecular_atmosphere_vs_particle_layer_{w.m}.png"
    outdir = os.path.join(artefact_dir, "plots")
    os.makedirs(outdir, exist_ok=True)
    fname_plot = os.path.join(outdir, filename)
    make_figure(fname_plot=fname_plot, brf_1=brf_1, brf_2=brf_2)

    outcome = np.allclose(brf_1.values, brf_2.values, rtol=5e-3)

    if outcome is False:
        print(f"Test failed, see artefact {fname_plot}")
        assert False
