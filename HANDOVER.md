# Handover — regression testing refactor + RAMI4ATM cleanup

Both branches merged into `tmp`. Commits `dc31090` … `2320a536`.
Working note, not a deliverable: drop before merge.

## Blocking

1. **Regeneration path unexercised.** The read path is verified (full suite
   green, see below); `--force-regen` has never written a reference. Check the
   diff is sane on one case:

   ```bash
   pytest tests/03_regression/romc/test_het01.py --force-regen --plot
   git -C resources/data diff --stat
   git -C resources/data checkout .         # discard
   ```

2. **RAMI4ATM Z-test thresholds** (`0.005`, and `0.05` for the m03 case, in
   `_toa_case`). Two effects push toward false failures: the smaller n loosens
   the Šidák correction, so each pixel is easier to reject; and until references
   are regenerated the Z-test finds no `radiance_srf_var` in them, warns, and
   falls back to the result variance alone — conservative by up to √2.
   Regenerate the references first, then re-run, then touch any threshold.

3. **Reference regeneration** — recommended, not required.

   ```bash
   pytest tests -m regression -k rami4atm --force-regen --plot \
       --artefact-dir <dir>
   git -C resources/data diff        # review, then commit in eradiate-data
   ```

   Not required because the tested variable names are unchanged. The references
   gain `radiance_srf_var`, which is what restores the full-precision Z-test;
   the BOA archived auxiliary variables are renamed (`radiance_srf1/2`,
   `radiosity_srf3/4` → `radiance_target`, `radiance_white`,
   `radiosity_target`, `radiosity_white`, each with a `_var` sibling).

   Same applies to `test_spherical_shell-ref.nc` after the move to
   `radiance_srf` (see the closed note below): it carries `radiance_srf` but
   only `radiance_var`, so the Z-test warns and falls back to the result
   variance until it is regenerated. The comment on the `1e-4` threshold also
   quotes a tail measurement taken on per-bin `radiance` (n = 1500); with
   `radiance_srf` n is 150 and the SRF integration over the 10 CKD bins should
   pull the tails back towards normal. Re-measure it the way the framework
   self-check constants were re-measured (see below).

## Decisions worth a second opinion

4. **Ocean tests: same criterion, different shape** — per-wavelength
   RMSE ≤ 1e-6, but one chart instead of N, with `w` mapped to hue. A failing
   wavelength surfaces as `details["worst w"]`.

5. **`radiosity` has no pipeline variance.** `postprocess_boa` derives one as
   the film-sum of `sector_radiosity_srf_var`, neglecting inter-pixel
   covariance, and says so in `_total_radiosity_var`. The alternative — a
   `radiosity_var` pipeline node — is a modelling decision worth making
   deliberately rather than as a side effect of this cleanup.

## Deferred, deliberately

6. **The BOA cases still use `RMSETest`** (`test_cases/rami4atm.py:547-548`).
   `postprocess_boa` emits `hdrf_var` and `bhr_var` that no criterion consumes;
   they are kept in the archived dataset because they are exactly what a future
   `ZTest` on `hdrf` would need.

## Closed since the original notes

- **Full suite green** — `pytest --plot` on 2026-08-05: 2299 passed, 7:24, exit
  0, `resources/data` clean afterwards. Covers every `tests/03_regression/**`
  case (RAMI4ATM ×33 incl. 2 BOA, ROMC het01/04/06, ocean ×3, atmospheres ×2,
  integrators ×2, spherical). `test_spherical` passes on `radiance_srf` against
  the un-regenerated reference, i.e. with the Z-test falling back to the result
  variance — the conservative direction.
- **Framework self-check constants re-measured** — the TODO at
  `test_regression_framework.py:44` is gone. On `radiance_srf` at spp = 1000
  (`ckd_double`, single realization): n = 76, per-pixel relative standard error
  0.60 % (0.33–0.81 % across viewing angles), paired difference 0.87 %. Family
  p-values by bias: 0 → 0.11, 1 % → 9e-4 (accepted), 2 % → 2e-6, 3 % → 5e-11,
  5 % → 6e-29, 10 % → 7e-110. Detection sets in just under 2 %, so `BIAS = 0.1`
  keeps a wide margin. Both `-m slow` tests pass in 11.7 s.
- **`test_report.py::test_robot_delegation` fixed** — it assigned to
  `ReportLogger._robot`, read-only since `30bc755b`. It now patches
  `robot.api.logger`, which the property imports at call time. Added
  `test_robot_autoselection`, covering the branch `30bc755b` introduced (no
  active Robot run ⇒ no backend); it swaps the module-level
  `EXECUTION_CONTEXTS` name rather than the real context stack, so a live
  Robot run is left alone. Whole file passes with and without
  `-p robotframework`.
- **Other suites still testing `radiance`** — resolved: only experiments with a
  band SRF are worth moving, since with a delta SRF `radiance_srf` is a
  degenerate copy. `spherical/test_spherical.py` (`sentinel_2a-msi-4`) was the
  only remaining one and now tests `radiance_srf`; RAMI4ATM was already moved.
  The rest keep `radiance` deliberately: `ocean/test_ocean_grasp.py`
  (`multi_delta`, 8 wavelengths), `romc/test_het01|04|06` (no `srf` → default
  `DeltaSRF(550 nm)`), `atmospheres/test_rpv_afgl1986[_continental]` and
  `integrators/test_eovolpath_{surface,canopy}` (`delta` @ 550 nm).
- **`has_atmoshphere` typo** — already spelled `has_atmosphere` throughout
  `test_cases/ocean.py`; the note was stale.
- **Unused test helpers removed** — `ocean_grasp_wavelength`
  (`test_cases/ocean.py`) and `session_timestamp`
  (`test_tools/fixtures/__init__.py`). No remaining references in the tree.
- **`bhr` could not be charted** — fixed by `2320a536`: `plot_x_axis()` holds
  out the VZA dimension only when the tested variable carries it, and
  single-point series are marked so they render.
- **`ReportLogger` dumped HTML fragments into pytest failure reports** outside a
  Robot run — fixed by `30bc755b`, backend now resolved at call time from the
  active Robot execution context.
- **`dim` aggregation for `ZTest`** — was an open question: slicing split the
  Šidák family into N families, each judged at `threshold`, so the overall
  false-positive rate grew as 1 − (1 − α)^N (40 % at α = 0.05 over ten
  wavelengths). Slices are now evaluated at 1 − (1 − α)^(1/N) and recombined
  with `sidak_family_p_value`, which makes `ZTest(dim=…)` verdict- and
  metric-identical to the un-sliced test — `dim` is diagnostic-only for
  Z-tests, and still a real tightening for `RMSETest`. Covered by
  `TestPerSliceEvaluation::test_ztest_matches_flattening`.
- **Docs build runs clean** — `pixi run -e docs docs` on 2026-08-05: build
  succeeded, 1 warning.
- **Robot report run exercised** — `pytest -p robotframework --plot -m regression
  tests/03_regression/romc/test_het01.py` on 2026-08-05: passes, `reports/log.html`
  written, no HTML fragment leakage. Note it overwrites `reports/`.
- **`--update-references` replaced** by `--force-regen`/`--regen-all`, with
  `--reference-dir` added (overrides both read and write locations). Reference
  file names are unchanged; no eradiate-data file needed renaming.
