# What's new

```{note}
For now, Eradiate uses a
"[Zero](https://0ver.org/)[Cal](https://calver.org/)Ver" versioning
scheme. The ZeroVer part reflects the relative instability of our API:
breaking changes may happen at any time. The CalVer part gives an idea of how
fresh the version you are running is. We plan to switch to a versioning
scheme more similar to SemVer in the future.

Updates are tracked in this change log. Every time you decide to update to a
newer version, we recommend that you to go through the list of changes.
```

% HEREAFTER IS A TEMPLATE FOR THE NEXT RELEASE
%
% ## vXX.YY.ZZ (unreleased)
%
% ### New features
%
% ### Breaking changes
%
% ### Deprecations
%
% ### Improvements and fixes
%
% ### Documentation
%
% ### Internal changes

## v0.22.3 (unreleased)

% ### New features

### Breaking changes

* Internal `_util` library is now `util.misc` ({ghcommit}`5a593d37b72a1070b5a8fa909359fd8ae6498d96`).

### Deprecations

* Deprecated function `ensure_array()` is removed ({ghcommit}`622821439cc4b66483518288e78dad0e9aa0da77`).

### Improvements and fixes

* Fix incorrect phase function blending in multi-component atmospheres ({ghpr}`197`).
* Fix incorrect volume data transform for spherical heterogeneous atmospheres ({ghpr}`199`).

% ### Documentation

### Internal changes

* The `progress` configuration variable is now an `IntEnum`, allowing for
  string-based setting while retaining comparison capabilities ({ghpr}`202`).

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
