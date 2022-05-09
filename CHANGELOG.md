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
% ## vXX.YY.ZZ (unreleased)
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

## v0.22.3 (unreleased)

### New features

* Add entry point `eradiate.run()`, which executes a full experiment pipeline 
  and returns results as an xarray dataset ({ghpr}`210`).

### Breaking changes

* Change the default value for the `ParticleLayer.dataset` field to:
  `spectra/particles/govaerts_2021-continental.nc` ({ghpr}`212`).

### Deprecations and removals

* Deprecated function `ensure_array()` is removed ({ghcommit}`622821439cc4b66483518288e78dad0e9aa0da77`).
* Deprecating instance method `Experiment.run()` ({ghpr}`210`).

### Improvements and fixes

* Add support for all missing AFGL 1986 reference atmospheres in CKD mode ({ghpr}`185`).
* Fix incorrect phase function blending in multi-component atmospheres 
  ({ghpr}`197`, {ghpr}`206`).
* Fix incorrect volume data transform for spherical heterogeneous atmospheres ({ghpr}`199`).
* Add default value for `CKDSpectralContext.bin_set` ({ghpr}`205`).
* Add a `-l` option to the `eradiate data info` command-line utility. If 
  this flag is set, the tool displays the list of files registered to each data 
  store ({ghpr}`208`).
* Add an optional `DATA_STORES` argument to the `eradiate data info` 
  command-line utility which may be used to select the data stores for which 
  information is requested ({ghpr}`208`).
* Add a new `load_dataset()` converter. It allows to set fields expecting an x
  array dataset using a path to a file or a data store resource ({ghpr}`212`).
* The `ParticleLayer` class no longer opens a dataset upon collision coefficient 
  evaluation; instead, its dataset field now holds an xarray dataset (instead 
  of a path), which does not change over the instance lifetime. 
  This reduces the amount of time spent on I/O ({ghpr}`212`).
* Optionally, export extra fields useful for analysis and debugging upon calling
  `AbstractHeterogeneousAtmosphere.eval_radprops()` ({ghpr}`206`, {ghpr}`212`).
* Re-formated `spectra/particles/govaerts_2021-*-extrapolated.nc` data sets
  ({ghpr}`213`).  
* `TabulatedPhaseFunction` interpolates input data on hundred times finer scattering angle cosine grid ({ghcommit}`2eb3408f1e249da353600e315af7ce09ca2f893f`).

### Documentation

* Add tutorials on homogeneous and molecular atmospheres ({ghpr}`194`).
* Added uninstall instructions ({ghpr}`217`).

### Internal changes

* The `progress` configuration variable is now an `IntEnum`, allowing for
  string-based setting while retaining comparison capabilities ({ghpr}`202`).
* Internal `_util` library is now `util.misc` ({ghcommit}`5a593d37b72a1070b5a8fa909359fd8ae6498d96`).
* Add a Numpydoc docstring parsing module ({ghpr}`200`).
* Add a `deprecated()` decorator to mark a component for deprecation ({ghpr}`200`).
* Update regression testing interface for improved robustness and ease of use 
  ({ghpr}`207`).
* Rewrite `eradiate data info` CLI for improved maintainability ({ghpr}`208`).

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
