# Release notes

```{note}
For now, Eradiate uses a [ZeroVer](https://0ver.org) versioning
scheme. It reflects the relative instability of our API: breaking changes may
happen at any time. We plan to switch to a versioning scheme more similar to
SemVer in the future.

Updates are tracked in this change log. Every time you decide to update to a
newer version, we recommend that you to go through the list of changes.
We try hard to remain backward-compatible and warn in advance for deprecation
when necessary—we also advise to not ignore `DeprecationWarning`s.

Emoji marks have the following meaning:

* ︎⚠ Requires particular attention during upgrade or usage
  (breaking change or experimental feature).
* 🗎 Documentation-related change.
* 🖳 This is a developer-facing change.
```

% HEREAFTER IS A TEMPLATE FOR THE NEXT RELEASE
%
% ## vXX.YY.ZZ (upcoming release)
%
% ### Deprecated
%
% ### Removed
%
% ### Added
%
% ### Changed
%
% ### Fixed
%

## v0.25.0 (upcoming release)

% ### Deprecated

% ### Removed

% ### Added

% ### Changed

% ### Fixed

* Set default units of the `*SpectralIndex.w` field to `ucc['wavelength']`
  ({ghpr}`362`).

## v0.24.0 (7th August 2023)

Release highlights:

* This is mainly a fix release. Thanks to our users for raising issues!
* An experimental interface to mesh-based preset canopies is added. The
  currently distributed data is limited to the Wellington Citrus Orchard scene;
  it will be expanded in the coming releases.

% ### Deprecated

% ### Removed

### Added

* ⚠ Added data files and APIs for the Wellington Citrus Orchard scene
  ({ghpr}`327`). This is an experimental feature, use with caution and report
  issues.

### Changed

* 🗎 Our change log now adheres [Keep a Changelog](https://keepachangelog.com).
* Our versioning scheme now adheres [ZeroVer](https://0ver.org).

### Fixed

* Added missing lookup strategy for participating medium parameters of the
  {class}`.HomogeneousAtmosphere` ({ghpr}`352`).
* Fixed a bug where the kernel dictionary emitted by the {class}`.MQDiffuseBSDF`
  wrapper would miss a data point on the azimuth dimension ({ghpr}`353`).
* Fixed a bug where an incorrect flag field init type would raise a `TypeError`
  on Python 3.11 ({ghpr}`358`).
* Fixed a bug where the init context could fall out of the range covered by the
  loaded spectral dataset ({ghpr}`360`).

## v0.23.2 (8th July 2023)

Release highlights:

* Eradiate is now published on PyPI and can be installed using Pip. See the
  [new installation instructions](https://eradiate.readthedocs.io/en/latest/rst/user_guide/install.html)
  This is still experimental and feedback is welcome.
* The spectral configuration of measures has changed. The new behaviour is
  documented in the
  [Spectral discretization guide](https://eradiate.readthedocs.io/en/latest/rst/user_guide/spectral_discretization.html).
* A new {class}`.AstroObjectIllumination` model has been added. It models the
  illumination by a distant celestial body with a finite apparent size in the
  sky. Support for this illumination model is currently experimental.
* The {class}`.DEMExperiment` now supports the spherical-shell geometry.
* The command-line interface now uses the Typer framework for improved user
  experience.

### Breaking changes

* Removed the Docker images implementation, deployment and documentation
  from the Eradiate repository ({ghpr}`322`). Docker builds are no longer
  supported for now.
* {class}`.Measure` no longer takes a `spectral_cfg` parameter but instead a
  `srf` parameter ({ghpr}`311`).
* In CKD modes with absorbing molecular atmospheres, the way to control the bin
  set is to set the atmosphere's `absorption_dataset` with an absorption
  dataset that has the corresponding desired bin set ({ghpr}`311`).
* {class}`.UniformSpectrum` is no longer a valid spectrum type to use for
  spectral response functions ({ghpr}`311`).
* The {meth}`.MultiDistantMeasure.from_viewing_angles` constructor is removed
  ({ghpr}`315`). Use one of the other {class}`.MultiDistantMeasure` constructors
  instead.
* The ``ertdata`` and ``ertshow`` command-line entry points are removed
  ({ghpr}`324`). Instead, use ``eradiate data`` and ``eradiate show``.
* The {class}`.KernelDictContext` class is renamed {class}`.KernelContext`
  ({ghpr}`324`).
* The `absorption_dataset` parameter of `AFGL1986RadProfile` and
  `US76ApproxRadProfile` is now required ({ghpr}`334`).

### Deprecations and removals

* ⚠ Removed {class}`.SpectralContext` and subclasses ({ghpr}`311`).
* ⚠ Removed {class}`.MeasureSpectralConfig` and subclasses ({ghpr}`311`).
* ⚠ Removed the {meth}`.MultiDistantMeasure.from_viewing_angles` constructor
  ({ghpr}`315`).
* ⚠ Removed ``ertdata`` and ``ertshow`` command-line entry points ({ghpr}`324`).

### Improvements and fixes

* Added {class}`.MultiDeltaSpectrum` spectrum type ({ghpr}`311`).
* Exposed several API members in the top-level namespace ({ghpr}`324`).
* Exposed the `eradiate.spectral.*` subpackage members in the
  {mod}`eradiate.spectral` namespace ({ghpr}`324`).
* Fixed incorrect Mitsuba scene parameter drop and lookup ({ghpr}`329`).
* Added spherical-shell geometry support to DEM components
  {ghpr}`320`.
* Fixed broken symmetry between {class}`.Spectrum` dictionary and object
  conversion protocols ({ghpr}`336`).
* Provided an Eradiate PyPI package ({ghpr}`328`).
* Added {class}`.AstroObjectIllumination` illumination type ({ghpr}`331`,
  {ghpr}`346`).
* Fixed a bug where {class}`.Layout` constructors would not raise if passed
  invalid azimuth values (*i.e.* outside the [0, 180]° range) ({ghpr}`345`).
* Added `wavelength_range` optional parameter to `MolecularAtmosphere.ussa_1976()`
  constructor to automatically open absorption datasets and restore working default
  constructor ({ghpr}`334`).

### Documentation

* Added a user guide page on spectral discretization.

### Internal changes

* Spectral dispatch with `functools.singledispatchmethod` ({ghpr}`311`).
* Experiments define the spectral discretization, based on information from
  its measure, its atmosphere components if applicable, and a default
  spectral set ({ghpr}`311`).
* Rewrote the CLI using the Typer framework ({ghpr}`326`).
* Refactored {class}`.Layout` constructor code ({ghpr}`345`).

## v0.23.1 (21 April 2023)

### Breaking changes

* The `coddington_2021-1_nm` dataset is now the default solar irradiance
  spectrum ({ghpr}`300`). If you used to work with the default value
  when defining the illumination in your experiments, this change might affect
  the measured radiance values.
* The newly introduced {class}`.DEMExperiment` now holds the support for digital
  elevation models. Accordingly, the {class}`.AtmosphereExperiment` does no longer
  support digital elevation models.

### Deprecations and removals

* The {meth}`.MultiDistantMeasure.from_viewing_angles()` constructor is
  deprecated and will be removed in v0.23.2.
* Removed {data}`.EradiateConfig.data_path` ({ghpr}`292`).
* {class}`.Measure`: Sample count split is retired ({ghpr}`296`).
* Removed the {class}`.AggregateSampleCount` pipeline step ({ghpr}`296`).
* {class}`.RadProfile`: The {class}`.ArrayRadProfile` class is retired
  ({ghpr}`296`).

### Improvements and fixes

* Added support for loading spectral response function data sets from custom
  paths ({ghpr}`270`).
* ⚠ Complete rewrite of the {class}`.MultiDistantMeasure` construction code
  ({ghpr}`274`, {ghpr}`281`). Previous functionality is preserved in the form of
  more specialized and simpler interfaces, and we now support a gridded coverage
  of the hemisphere specified as the Cartesian product of zenith and azimuth
  lists of values.
* Added a helper to grid against VZA and VAA the results obtained with a
  {class}`.MultiDistantMeasure` with a grid layout ({ghpr}`274`). See
  {meth}`~eradiate.xarray.unstack_mdistant_grid`.
* `MeasureSpectralConfig.srf`'s converter loads the prepared SRF version first,
  by default, and falls back to the raw version if the former does not exist,
  in the case where `srf` is specified by a keyword ({ghpr}`278`).
* Fixed the behaviour of the {class}`.MeshTreeElement` constructor when no units
  are specified ({ghpr}`279`).
* Extended the distant measure line with the possibility to control the
  distance between ray origins and the target ({ghpr}`275`). The default
  behaviour is unchanged, effectively positioning ray origins at an infinite
  distance from the target.
* All measures can now be attached a non-default sampler ({ghpr}`280`).
* Fixed unnecessary memory allocations ({ghpr}`282`).
* Added utility functions to `srf_tools` module ({ghpr}`283`).
* Added {class}`.MQDiffuseBSDF` reflection model ({ghpr}`286`).
* Fixed a bug where the `bilambertian` BSDF plugin would produce incorrect
  results when used with LLVM Mitsuba variants ({ghpr}`297`).
* ⚠ Refactoring of the {class}`.Mode` infrastructure ({ghpr}`298`).
* Added versions 1 and 2 and the full spectrum extension of the
  TSIS-1 HSRS solar irradiance spectra ({ghpr}`300`).
* The `coddington_2021-1_nm` dataset is now the default solar irradiance
  spectrum ({ghpr}`300`).
* Introduced the {class}`.DEMExperiment` to handle scenes with digital elevation
  models. At this point it only supports plane-parallel atmospheric geometries
  ({ghpr}`289`).
* ⚠ Rewrite of the kernel interface ({ghpr}`296`).
* ⚠ Updates to the scene type hierarchy ({ghpr}`296`).
* ⚠ All measures are now batch-computed at each iteration of the spectral loop
  ({ghpr}`296`).
* ⚠ {class}`.Atmosphere` type hierarchy updates: altitude grid control, common
  spectral evaluation interface ({ghpr}`296`).
* {class}`.BlendPhaseFunction` code was transitioned from a recursive to an
  iterative loop-based implementation ({ghpr}`296`).
* {class}`.RadProfile` evaluation on arbitrary altitude grids is now permitted
  ({ghpr}`296`).
* Introduced the {class}`.SpotIllumination`, which points a beam of light of
  fixed angular width at a target location ({ghpr}`302`).
* Added a new module {mod}`eradiate.constants` to store physical constants used
  in Eradiate ({ghpr}`312`).
* Added absorption and scattering bypass switches to the {class}`.ParticleLayer`
  class ({ghpr}`316`).
* ⚠ Moved Mitsuba logs to the `mitsuba` logger ({ghpr}`318`).
* ⚠ Centralized geometric information to `SceneGeometry` ({ghpr}`319`).
* ⚠ Made {class}`Atmosphere`'s `_params_*` properties abstract for improved
  safety ({ghpr}`319`).
* Fixed a major issue in volume definitions and parameter updates of
  {class}`BlendPhaseFunction` ({ghpr}`319`).

### Documentation

* Upgraded Sphinx Book theme to v1.0.0 ({ghpr}`306`).

### Internal changes

* Updated Mitsuba submodule to v3.2.1 ({ghpr}`277`, {ghpr}`296`).
  This notably fixes a
  [k-d tree creation issue](https://github.com/mitsuba-renderer/mitsuba3/issues/233),
  [missing initialization code in the `blendphase` plugin](https://github.com/mitsuba-renderer/mitsuba3/issues/488)
  and [an incorrect setter for volume data containers](https://github.com/mitsuba-renderer/mitsuba3/issues/480).
* Harmonize dataset converters for solar irradiance spectra, spectral response
  function and particle radiative property datasets ({ghpr}`284`).
* Replaced isort with [Ruff](https://github.com/charliermarsh/ruff) ({ghpr}`299`).
* Changed data types of data variables in spectral response function datasets
  from double to single floating point numbers ({ghpr}`300`).
* Added TOML formatting pre-commit hook ({ghpr}`305`).
* Updated dependency management system to latest tooling changes ({ghpr}`306`).
* Add {func}`.cache_by_id` to replace `@functools.lru_cache(maxsize=1)` when
  appropriate ({ghpr}`315`).
* Clarify particle layer optical thickness computation ({ghpr}`321`).

## v0.22.5 (17 October 2022)

### New features

* Added support for various azimuth definition conventions ({ghpr}`247`).
* Added an offline mode which disables all data file downloads ({ghpr}`249`).
* Added support for Digital Elevation Models (DEM) ({ghpr}`246`).

### Breaking changes

* Dropped the xarray metadata validation system ({ghpr}`266`).

### Deprecations and removals

* Removed the deprecated `Experiment.run()` method ({ghpr}`210`).
* Removed the path resolver component ({ghpr}`251`).
* Renamed `Experiment` classes ({ghpr}`252`).

    * `OneDimExperiment` ⇒ `AtmosphereExperiment`
    * `RamiExperiment` ⇒ `CanopyExperiment`
    * `Rami4ATMExperiment` ⇒ `CanopyAtmosphereExperiment`

### Improvements and fixes

* Added the atmo-centimeter, a.k.a. atm-cm, to the unit registry ({ghpr}`245`).
* Added spectral response functions for the VIIRS instrument onboard JPSS1 and
  NPP platforms ({ghpr}`253`).
* Submodules and packages are now imported lazily ({ghpr}`254`, {ghpr}`261`).
  This significantly decreases import time for most use cases.
* Optimized calls to `Quantity.m_as()` in
  `InstancedCanopyElement.kernel_instances()` ({ghpr}`256`).
* Fixed incorrect scaling formula for datetime-based scaling of Solar irradiance
  spectra ({ghpr}`258`).
* Added system information report to the `eradiate show` command-line utility
  ({ghpr}`264`).
* Some dependencies are now optional, although recommended ({ghpr}`266`).
* Added new sahara and continental particle radiative properties including the
  full scattering phase matrix ({ghpr}`259`).
* Allow mixing a purely absorbing `MolecularAtmosphere` with a `ParticleLayer`
  ({ghpr}`239`).
* Change the `ExponentialParticleDistribution` formulation to rate-based; allow
  scale-based parametrization for initialization ({ghpr}`271`).
* Fixed missing parameters in the `LeafCloud.sphere()` constructor
  ({ghpr}`272`).
* Add spectral response function filtering utility ({ghpr}`269`).

### Documentation

* Dependencies are now listed explicitly ({ghpr}`266`).
* Major tutorial overhaul ({ghpr}`209`).
* Rendered tutorial notebooks and added thumbnail galleries ({ghpr}`273`).

### Internal changes

* Updated Mitsuba submodule to v3.0.2 ({ghpr}`250`, {ghpr}`255`, {ghpr}`267`).
  This notably fixes the sampling method of the `tabphase` plugin and a Dr.Jit
  warning on some Linux machines.
* Aligned the `tabphase_irregular` plugin with the fix `tabphase` code
  ({ghpr}`255`).
* Updated codebase to use ``attrs``'
  [next-generation APIs](https://www.attrs.org/en/stable/names.html#tl-dr)
  ({ghpr}`268`).

---

## v0.22.4 (17 June 2022)

### New features

* Added a `InterpolatedSpectrum.from_dataarray()` class method constructor
  ({ghpr}`243`).

### Breaking changes

* Removed temporary volume files used by the `AbstractHeterogeneousAtmosphere`
  line and the `BlendPhaseFunction` class (replaced by in-memory buffers).
  Corresponding file name and cache directory control parameters were removed as
  well ({ghpr}`231`).

### Deprecations and removals

* Updated all tests to use `eradiate.run()` instead of the deprecated
  `Experiment.run()` method ({ghpr}`227`).

### Improvements and fixes

* Added spectral response function data sets for POLDER instrument onboard
  PARASOL platform ({ghpr}`232`).
* Added an extrapolated version of the ``thuillier_2003`` solar irradiance
  spectrum to cover the full wavelength range of Eradiate ({ghpr}`233`).
* Fixed a bug where converting an integer to a `Spectrum` would fail
  ({ghpr}`236`).
* Fixed a bug where the two BSDFs in the `CentralPatchSurface` would be
  assigned in the wrong order ({ghpr}`237`).
* Raise an exception when molecular concentrations are out of bounds
  ({ghpr}`237`).
* Various fixes to the `rpv` plugin, among which a missing PDF term in the
  `sample()` method ({ghpr}`240`).
* Fix incorrect spectral indexing of result datasets in CKD mode ({ghpr}`241`).
* Improve the Solar irradiance spectrum initialization sequence ({ghpr}`242`).
* The `thuillier_2003_extrapolated` dataset is now the default Solar irradiance
  spectrum ({ghpr}`242`).

### Internal changes

* Transitioned dependency management to `pyproject.toml` following
  [PEP 621](https://peps.python.org/pep-0621/) ({ghpr}`203`).
* Updated Mitsuba submodule ({ghpr}`228`).
* Removed the angular regridding feature in `TabulatedPhaseFunction` and
  replaced with plugin selection ({ghpr}`229`).

---

## v0.22.3 (22 May 2022)

### New features

* Added entry point `eradiate.run()`, which executes a full experiment pipeline
  and returns results as an xarray dataset ({ghpr}`210`).

### Breaking changes

* Changed the `ParticleLayer.dataset` field's default value to the more useful
  the continental aerosol dataset
  `spectra/particles/govaerts_2021-continental.nc` ({ghpr}`212`).
* Changed the interface of the `ParticleLayer`: the `tau_550` field is replaced
  by a more general `tau_ref` which sets the extinction optical thickness of the
  particle layer at a reference wavelength specified by the `w_ref` field.
  `w_ref` defaults to 550 nm, thus preserving prior behaviour ({ghpr}`221`).

### Deprecations and removals

* Removed deprecated function `ensure_array()`
  ({ghcommit}`622821439cc4b66483518288e78dad0e9aa0da77`).
* Deprecated instance method `Experiment.run()` ({ghpr}`210`).

### Improvements and fixes

* Added support for all missing AFGL 1986 reference atmospheres in CKD mode
  ({ghpr}`185`).
* Fixed incorrect phase function blending in multi-component atmospheres
  ({ghpr}`197`, {ghpr}`206`).
* Fixed incorrect volume data transform for spherical heterogeneous atmospheres
  ({ghpr}`199`).
* Added default value for `CKDSpectralContext.bin_set` ({ghpr}`205`).
* Added a `-l` option to the `eradiate data info` command-line utility. If
  this flag is set, the tool displays the list of files registered to each data
  store ({ghpr}`208`).
* Added an optional `DATA_STORES` argument to the `eradiate data info`
  command-line utility which may be used to select the data stores for which
  information is requested ({ghpr}`208`).
* Added a new `load_dataset()` converter. It allows to set fields expecting an x
  array dataset using a path to a file or a data store resource ({ghpr}`212`).
* The `ParticleLayer` class no longer opens a dataset upon collision coefficient
  evaluation; instead, its dataset field now holds an xarray dataset (instead
  of a path), which does not change over the instance lifetime.
  This reduces the amount of time spent on I/O ({ghpr}`212`).
* Added the possibility to optionally export extra fields useful for analysis
  and debugging upon calling `AbstractHeterogeneousAtmosphere.eval_radprops()`
  ({ghpr}`206`, {ghpr}`212`).
* Re-formatted `spectra/particles/govaerts_2021-*-extrapolated.nc` data sets
  ({ghpr}`213`).
* Replaced leftover calls to deprecated `eradiate.data.open` with
  `eradiate.data.open_dataset` ({ghpr}`220`).
* Improved `TabulatedPhaseFunction`'s behaviour. If the phase function lookup
  table coordinate `mu` defines a regular grid, the phase function is no longer
  resampled on a regular grid, which results in improved performance. Otherwise,
  *i.e.* if `mu` defines an irregular grid, phase function data is resampled on
  a regular `mu` grid with a step equal to the smallest detected step in the
  `mu` coordinate array, which preserves accuracy ({ghpr}`226`).

### Documentation

* Added tutorials on homogeneous and molecular atmospheres ({ghpr}`194`).
* Added uninstallation instructions ({ghpr}`217`).
* `ParticleLayer`: document format of `dataset` attribute ({ghpr}`223`).

### Internal changes

* The `progress` configuration variable is now an `IntEnum`, allowing for
  string-based setting while retaining comparison capabilities ({ghpr}`202`).
* Internal `_util` library is now `util.misc`
  ({ghcommit}`5a593d37b72a1070b5a8fa909359fd8ae6498d96`).
* Added a Numpydoc docstring parsing module ({ghpr}`200`).
* Added a `deprecated()` decorator to mark a component for deprecation
  ({ghpr}`200`).
* Updated regression testing interface for improved robustness and ease of use
  ({ghpr}`207`).
* Rewrote `eradiate data info` CLI for improved maintainability ({ghpr}`208`).
* Refactored `ParticleLayer` unit tests and added system tests
  ({ghpr}`219`, {ghpr}`222`, {ghpr}`224`).

---

## v0.22.2 (23 March 2022)

### New features

* Added IPython extension ({ghcommit}`759c7e7f8a446f00a737f095f0cf5261c350b8d5`,
  {ghcommit}`c3b30c9f38298712ab5697fc0a7a37fa39b8cdbf`).

### Improvements and fixes

* Account for spectral dependency of the King correction factor ({ghpr}`187`).
* Fix wrong atmosphere shape size in plane parallel geometry with no scattering ({ghpr}`195`).

### Documentation

* Major update of all documentation contents ({ghpr}`192`, {ghpr}`193` and many commits).
* Add Sphinx roles for GitHub links ({ghcommit}`2789499b3ba66f2b00c1a0987fdaa9cdc1f5f705`).
* Add SVG to PNG export script ({ghcommit}`9e942a48ae69b076c3f70d589dd6cd8c6580b563`).
* Update logos with a more modern style ({ghcommit}`a3dfe36fd9b2f7e5842c60a7d5204f9d2138072e`).

### Internal changes

* Refactor regression testing framework to handle more use cases and make it
  more robust ({ghpr}`188`).

---

## v0.22.1 (14 March 2022)

This is the first official release of Eradiate.
