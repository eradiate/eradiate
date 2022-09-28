# What's new?

```{note}
For now, Eradiate uses a
"[Zero](https://0ver.org/)[Cal](https://calver.org/)Ver" versioning
scheme. The ZeroVer part reflects the relative instability of our API:
breaking changes may happen at any time. The CalVer part gives an idea of how
fresh the version you are running is. We plan to switch to a versioning
scheme more similar to SemVer in the future.

Updates are tracked in this change log. Every time you decide to update to a
newer version, we recommend that you to go through the list of changes.
We try hard to remain backward-compatible and warn in advance for deprecation
when necessary—we also advise to not ignore `DeprecationWarning`s.
```

% HEREAFTER IS A TEMPLATE FOR THE NEXT RELEASE
%
% ## vXX.YY.ZZ (upcoming release)
%
% ### New features
%
% ### Breaking changes
%
% ### Deprecations and removals
%
% ### Improvements and fixes
%
% ### Documentation
%
% ### Internal changes

## v0.22.5 (upcoming release)

### New features

* Added support for various azimuth definition conventions ({ghpr}`247`).
* Added an offline mode which disables all data file downloads ({ghpr}`249`).
* Added support for Digital Elevation Models (DEM) ({ghpr}`246`).

### Breaking changes

* Dropped the xarray metadata validation system ({ghpr}`266`).

### Deprecations and removals

* Removed the path resolver component ({ghpr}`251`).
* Renamed `Experiment` classes ({ghpr}`252`).

  * `OneDimExperiment` ⇒ `AtmosphereExperiment`
  * `RamiExperiment` ⇒ `CanopyExperiment`
  * `Rami4ATMExperiment` ⇒ `CanopyAtmosphereExperiment`

### Improvements and fixes

* Added the atmo-centimeter, a.k.a. atm-cm, to the unit registry ({ghpr}`245`).
* Added spectral response functions for the VIIRS instrument onboard JPSS1 and
  NPP platforms ({ghpr}`253`).
* Submodules and packages are now imported lazily ({ghpr}`254`). This
  significantly decreases import time for most use cases.
* Optimised calls to `Quantity.m_as()` in
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

### Documentation

* Dependencies are now listed explicitly ({ghpr}`266`).

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
* Improve the Solar irradiance spectrum initialisation sequence ({ghpr}`242`).
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
