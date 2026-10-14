---
name: merge-upstream-mitsuba
description: Merge a new upstream Mitsuba 3 release into Eradiate's kernel submodule (ext/mitsuba), resolve conflicts while preserving Eradiate changes, rebuild the kernel, fix plugin breakage from Mitsuba/Dr.Jit API changes, bump kernel versions and update the parent Eradiate repo. Use whenever the user wants to upgrade, update, sync or rebase the kernel onto a Mitsuba version (e.g. "bring ext/mitsuba to v3.10", "merge upstream mitsuba", "upgrade the kernel to the latest Mitsuba release"), even if they do not say "merge".
---

# Merge an upstream Mitsuba release into the Eradiate kernel

The kernel in `ext/mitsuba` is a fork of Mitsuba 3. Eradiate changes to
upstream files sit between `ERADIATE_CHANGE_BEGIN`/`ERADIATE_CHANGE_END`
fences; Eradiate-only code lives in `src/eradiate_plugins/`,
`include/mitsuba/render/eradiate/` and similar `eradiate` directories. An
upgrade is a real merge commit (not a rebase), followed by separate fix-up
commits, so that history shows exactly what upstream brought and what we had
to adapt.

The work spans two repositories. Keep track of which one each command runs in.

## 0. Gather inputs and confirm

Ask (or infer from the request) and confirm before changing anything:

- **Upstream ref**: usually a local `upstream` branch or a tag (`vX.Y.Z`)
  in `ext/mitsuba`. Check it with `git describe --tags <ref>`.
- **Kernel base branch**: `main` in `ext/mitsuba`, unless told otherwise.
- **Parent base branch**: `next` in the Eradiate repo, unless told otherwise.
- **New kernel patch version** (`ERADIATE_MI_VERSION`, minor bump by default).

Branch names follow `merge_<version without dots>` (e.g. `merge_391`) in
both repositories. Stop at local commits; push only when the user asks.

## 1. Survey the merge (read-only)

In `ext/mitsuba`:

```shell
git merge-base <base> <upstream>
git rev-list --count <base>..<upstream>     # incoming upstream commits
git merge-tree --write-tree --name-only <base> <upstream>   # conflict list
git diff <merge-base> <upstream> -- pyproject.toml .gitmodules include/mitsuba/mitsuba.h
```

Note the Dr.Jit/nanobind version bumps and submodule additions/removals:
they predict where the build will break.

## 2. Merge and resolve

```shell
git switch -c merge_<ver> <base>
git merge --no-ff <upstream> -m "Merge tag 'v<X.Y.Z>' into merge_<ver>"
```

Resolution rules, learned from past merges:

| File | Resolution |
| --- | --- |
| `.gitmodules` | Union: upstream entries plus ours (`resources/data_eradiate`) |
| `CMakeLists.txt` | Keep ours where it conflicts (Embree OFF, mono `MI_DEFAULT_VARIANTS`); take the rest of upstream |
| `README.md` | Keep ours |
| `pyproject.toml` | Keep our package metadata (`eradiate-mitsuba`, setuptools-scm version override, `--preset=eradiate`); take upstream's Dr.Jit/nanobind pins and cibuildwheel changes |
| `.github/workflows/wheels.yml` | Keep our version override and variant handling (variants come from the CMake preset, so drop upstream's `VARIANTS` env); take upstream's toolchain updates (cibuildwheel, stable-ABI setup). Settle the build matrix in step 5 |
| Upstream C++ files | Take upstream's structure, then re-apply the code inside our fences |
| `include/mitsuba/python/docstr.h` | Generated: run `scripts/merge_docstr.py` now (see below), regenerate in step 6 |

For C++ conflicts, before re-applying a fenced block, check whether upstream
now contains the same fix: Eradiate sometimes backports upstream PRs, and
keeping both copies gives duplicate definitions. Compare the fenced code to
`git show <upstream>:<file>`. When a definition moved to another file
upstream (e.g. `ShapeType` moved from `shape.h` to `fwd.h`), move our
additions with it, inside a fence.

**docstr.h**: never resolve it hunk by hunk. Git's hunks cut through
multi-line docstrings, so hunk-wise resolution silently drops or duplicates
declarations. Run, from `ext/mitsuba`:

```shell
python <skill-dir>/scripts/merge_docstr.py <merge-base> <base> <upstream>
```

Check that no conflict markers remain (`git grep -nE '^(<<<<<<<|>>>>>>>) '`),
that `include/mitsuba/mitsuba.h` carries the new `MI_VERSION_*` and the
intact `ERADIATE_MI_VERSION` block, then stage everything.

```shell
git submodule sync --recursive
git submodule update --init --recursive
```

Old submodule directories that upstream removed or renamed stay behind as
untracked directories; leave them and mention them in the report.

`git submodule update` skips a submodule already at the pinned commit, even
if its working tree files were deleted during the merge (seen with
`ext/rgb2spec`; CMake then fails with "does not contain a CMakeLists.txt
file"). List submodules with local changes and restore deleted files:

```shell
git submodule foreach --recursive --quiet \
  'git status --porcelain | grep -q . && echo "$displaypath" || true'
git -C <path> checkout -- .  # for each listed submodule with only deletions
```

**Commit the merge.** The `eradiate-change-fences` pre-commit hook flags
every upstream line in a merge, so skip it for the merge commit only, after
checking fence pairing on the files you edited by hand:

```shell
python tools/check_eradiate_fences.py <files you edited>
SKIP=eradiate-change-fences git commit --no-edit
```

## 3. Build and fix

Run from the parent repo. Upstream template changes (`resources/mitsuba.conf.template`)
and submodule restructuring make an incremental build unreliable, so do a
clean build. `kernel-clean` deletes `build/mitsuba.conf`: back it up first and
compare its `enabled` variants with the regenerated one afterwards.

```shell
cp ext/mitsuba/build/mitsuba.conf <scratch>/mitsuba.conf.bak
pixi run kernel-clean
pixi run kernel-build          # configure + build
pixi run cmake --build ext/mitsuba/build -- -k 0   # after the first failure
```

Use `-k 0` once the first build fails so ninja reports all broken targets in
one pass, then group errors by file. Builds take tens of minutes: run them in
the background. In zsh, overwrite logs with `>|` (noclobber is set).

Fix breakage in Eradiate code by copying how upstream adapted its own
equivalent code (e.g. our `arectangle` vs upstream `rectangle`/`disk`, our
`measured_mono` vs upstream `measured`). Look at `git diff <merge-base>
<upstream> -- <upstream equivalent>` for the pattern. Examples from past
upgrades:

- removed Dr.Jit macros (`DRJIT_STRUCT_NODEF` became
  `DRJIT_ARRAY_DEFAULTS(Name)` + `DRJIT_TRAVERSE(Name, ...)`);
- texture evaluation returning values instead of writing to pointers
  (`tex.eval<dr::Array<Float, 1>>(p, active).x()`);
- `ray_intersect_preliminary_impl` returning a leading validity mask;
- `Struct::Type` replaced by `sj::Type` (struct-jit).

Commit these as `Post-merge fixes (upgrade to v<X.Y.Z>)`, with a bullet per
change in the body.

## 4. Kernel tests

```shell
pixi run pytest ext/mitsuba/src/eradiate_plugins/tests -q
```

For every failure, decide whether the merge caused it:

- **Upstream behaviour change** (e.g. `blendphase` returning the mixture pdf,
  grid volume min/max propagating NaN): adapt the test or plugin to the new
  behaviour and explain it in the post-merge commit.
- **Already broken before the merge**: check whether the test and the code it
  exercises came from parallel branches
  (`git merge-base --is-ancestor <commit A> <commit B>`). Fix it in its own
  commit so the post-merge commit stays about the upgrade.

The plugin test files have pre-existing ruff violations; if the hooks block a
commit for violations you did not introduce, revert the hooks' reformatting
and commit with `SKIP=ruff-check,ruff-format,trailing-whitespace`. Report it.

## 5. Wheel build matrix

Our wheels depend on the Dr.Jit wheel pinned in `pyproject.toml`, so a
platform or Python version without a matching Dr.Jit wheel cannot be
installed. Upstream Mitsuba aligns its own matrix with the Dr.Jit version it
tracks, so the merged upstream matrix is a good starting point; the Dr.Jit
release remains the reference:

- `.github/workflows/wheels.yml` matrix: same platforms as Dr.Jit's wheel
  workflow (for the pinned tag), same Python range minus end-of-life
  versions. Pin the Windows runner to a named image rather than
  `windows-latest`, so the toolchain does not drift.
- `pyproject.toml`: `requires-python` and the cibuildwheel `build` list
  must match the matrix.

Our matrix must be a subset of Dr.Jit's. Check it from `ext/mitsuba`, with
`ext/drjit` checked out at the pinned release (after the submodule update in
step 2):

```shell
python <skill-dir>/scripts/check_ci_matrix.py
```

It compares platforms (runner labels mapped to OS and architecture, since
both projects may pin different images for the same platform), Python tags,
and cibuildwheel `build` lists, and checks that our `build` list agrees with
our own matrix. Fix every reported failure before committing. If it reports
an unknown runner label, extend `runner_platform()` in the script.

Commit as `Update the CI build matrix`, stating the platforms or Python
versions dropped and why.

## 6. Regenerate docstrings and API docs

The docstr.h merge from step 2 only produces a header that compiles; the
docstrings still describe the pre-merge Eradiate headers. Regenerate them
once the build passes, from `ext/mitsuba` (it has its own pixi workspace):

```shell
pixi run docstrings
pixi run docs-clean && pixi run docs-api
```

`docs-clean` matters: on an incremental Sphinx build the autodoc callbacks
do not fire and `docs_eradiate/generated/` is not rewritten (see
`docs_eradiate/dev_guide/api_docs.rst`). Commit `docstr.h` and the changes
under `docs_eradiate/` as `Update docstrings and API docs`.

## 7. Bump versions

- Kernel version: `ERADIATE_MI_VERSION_*` in
  `ext/mitsuba/include/mitsuba/mitsuba.h`, managed by
  `ext/mitsuba/.bumpversion.toml`. First check that its `current_version`
  matches the header (it has drifted before), then run
  `uvx bump-my-version bump minor` from `ext/mitsuba` and commit as
  `Bump eradiate-mitsuba version`. This header is included everywhere, so a
  rebuild follows.
- Parent `src/eradiate/kernel/_versions.py`: `REQUIRED_MITSUBA_VERSION` and
  `REQUIRED_MITSUBA_PATCH_VERSION` (checked against `mitsuba.MI_VERSION` and
  `mitsuba.ERADIATE_MI_VERSION` at import).
- Parent `pyproject.toml` `kernel` extra (`eradiate-mitsuba>=...`): leave it
  unless the new wheel is already on PyPI. Pixi solves all environments,
  including ones with the `kernel` extra, so an unpublished version breaks
  every `pixi run`. Tell the user to raise it at release time.
- Links to versioned Mitsuba docs in `docs/` (grep for the old version).

## 8. Parent repo and Eradiate tests

```shell
git switch -c merge_<ver> <parent base>
git add src/eradiate/kernel/_versions.py ext/mitsuba
git commit -m "Upgrade kernel to Mitsuba v<X.Y.Z>"
pixi run test
```

Do not stage `resources/data` or other submodules the user has modified
locally. Test runs may leave Robot Framework artefacts (`log.html`,
`output.xml`, `report.html`) in the repo root: delete the ones your runs
created.

## 9. Report

Summarise for the user:

- commits in each repository, with hashes;
- conflict resolutions that involved judgement (dropped duplicates, moved
  code, wheels.yml choices, docstr.h script output);
- API fixes and behaviour changes adopted from upstream;
- test results, separating merge-caused failures from pre-existing ones;
- the CI matrix check result and any platforms or Python versions dropped;
- skipped hooks, unchanged version pins, leftover untracked directories,
  and anything else left for the user to do (pushing: kernel first, since
  the parent's submodule pointer must resolve).
