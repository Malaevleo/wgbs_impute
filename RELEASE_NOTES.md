# 0.2.0

- Package installation and console entry points replace parent-directory/PYTHONPATH setup.
- Read thinning is internal; no imports from research scripts or user directories remain.
- Repeated sample IDs across distinct chromosome files form one reference library.
- Ambiguous filename identities and overlapping reference files fail explicitly.
- Disease-role filters are applied before files enter a same-directory panel.
- Standard Bismark coverage coordinates are normalized from 1-based to 0-based. `.zero.cov` is also supported. Rebuild panels derived from Bismark files.
- Malformed rows and nonfinite, negative, fractional, or out-of-range read counts fail with line-number diagnostics instead of being silently discarded or clipped.
- Query chromosomes absent from the panel are preserved. `--include-panel-sites` adds absent query positions as coverage-zero rows when requested.
- Sample identity is inferred for exclusion, with an explicit override for renamed files. Fusion calibration is refreshed when the reference selection or fitting parameters change.
- Missing predictions are omitted from numeric BED output and retained in a separate table. They are never replaced with zero methylation.
- Covariance matrices are fitted on demand. The large per-chromosome covariance tensor is no longer stored or allocated. Output and summary rows are streamed.
- Cache schema 2 validates completeness/version, prevents concurrent writers, and publishes a cache only after all preparation succeeds. Existing caches are never overwritten.
- Run reports record effective arguments, selected references per chromosome, and manifest provenance.
- The Docker image installs the distributable package, pins runtime dependencies, runs unprivileged, and supports offline/read-only execution with writable temporary/output mounts.
- Portable numerical, parser, manifest, cache, reporting, and container regression tests replace machine-dependent research imports. GitHub Actions checks installation, packaging, lint, and Docker execution.

## Upgrade

Install the new package, create new cache and output directories, and regenerate the panel from original reference files. For renamed queries, supply `--sample-id`. Callers that previously relied on malformed rows being skipped must correct those inputs. Python 3.10+ is required.

## Local release validation (2026-10-29)

- 29 regression tests passed against an installed wheel on macOS ARM64 / Python 3.12, with the working directory outside the source checkout.
- The same 29 tests passed inside the Linux ARM64 / Python 3.11 Docker image with networking disabled, an unprivileged user, and a read-only root filesystem.
- The offline Docker demo wrote 400 BED rows through a host-mounted output directory. Output counts and the completion marker were checked. Synthetic fused MAE was 4.113 percentage points, versus 8.420 for the panel mean.
- A smoke run on existing chromosome-21 data processed 21,944 query sites, preserved high-coverage observations, and kept every available prediction within 0–100%. This checks execution and output invariants, not held-out scientific accuracy.
- Ruff lint/format checks, source/wheel builds, and installed dependency checks passed. The GitHub Actions matrix is configured for Python 3.10–3.13 and will run after pushing; those remote checks have not run locally.
