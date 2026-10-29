# wgbs_impute

Reference-panel imputation for low-coverage CpG methylation. Fused BLUP predicts a site's methylation from nearby CpGs and combines that prediction with the query's own reads. Reference and query coordinates must use the same genome assembly and chromosome names. No organism, cohort, tissue, or filesystem path is built into the algorithm.

## Install

Python 3.10 or newer:

```bash
git clone https://github.com/Malaevleo/wgbs_impute.git
cd wgbs_impute
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
wgbs-impute --help
```

Commands work from any directory after installation. NumPy and threadpoolctl are the core dependencies. Install `python -m pip install '.[ml]'` to enable the optional CatBoost `local` and `global` methods, or `python -m pip install -r requirements.txt` for all runtime dependencies before installing the package.

## Quickstart

```bash
wgbs-impute make-manifest --panel-dir references/ --output manifest.tsv
wgbs-impute prepare-panel --manifest manifest.tsv --cache panel_cache/
wgbs-impute impute --query query.bed.gz --panel-cache panel_cache/ --output query.imputed.tsv
```

The equivalent `python -m wgbs_impute` commands also work. Panel preparation refuses an existing cache directory. Use a new directory to rebuild a panel. A cache appears at its final path only after preparation completes successfully; concurrent writers are prevented by a sibling `.lock` file. After a forcibly terminated process, verify that it has stopped before removing its leftover lock file.

By default, only rows present in the query are written, including zero-coverage rows. To also impute panel sites absent from the query (common with Bismark coverage files):

```bash
wgbs-impute impute --query query.cov.gz --panel-cache panel_cache/ \
  --output query.imputed.tsv --include-panel-sites
```

This adds sites on every cached chromosome, or just `--chrom` when specified. It can increase output size substantially. CpGs outside both the query and the panel cannot be invented. Unmatched query sites and chromosomes are preserved as `passthrough`; unobserved sites with no prediction remain missing.

## Reference manifests and sample identity

`make-manifest` scans one directory, without recursion, for `.bed`, `.bed.gz`, `.cov`, and `.cov.gz`. The manifest columns are `path`, `sample_id`, `group`, and `chrom`. Only `path` and `sample_id` are required. Relative paths are resolved against the manifest's directory. `group` labels are user-defined. Optional `mean_cov` overrides the measured mean positive coverage; otherwise it is computed across all observed sites, weighted by the number of sites.

Filename inference takes the token before the first `_` or `.`, so `SAMPLE_CellType.chr1.bed.gz` becomes `SAMPLE`. Use an explicit manifest when your naming scheme differs. Chromosome-split files may share a sample ID when their `chrom` fields differ; they are merged into one library. Duplicate identities on the same chromosome, overlapping files, and conflicting metadata fail explicitly. When `chrom` is supplied, only that chromosome is read into the panel. Whole-genome files leave `chrom` blank.

The query's inferred sample ID is excluded from the panel automatically when it matches. **Pass `--sample-id` explicitly if the query was renamed**, including when mounted at a generic container path. Exclusion applies to every chromosome. Selecting fewer than two references after exclusion is an error. `--n-refs` selects the deepest libraries while balancing across the supplied groups; zero uses the full panel.

Disease libraries can be added from a separate directory:

```bash
wgbs-impute make-manifest --panel-dir references/ --disease-dir disease/ --output manifest.tsv
```

An optional `samples.tsv` in that directory or its parent has `sample_id` (or `epirr`), `disease`, and optional `role` columns. A missing role defaults to `ref`. When a table is present, unlisted samples are excluded, and only the requested role is admitted (default `ref`; `--disease-role target` or `all` changes this). Disease names become group labels. Three-column caches named `SAMPLE.chr1.tsv` are also accepted from the disease directory. Passing the same directory as both panel and disease input applies the table once, before admitting any libraries.

## Input formats and validation

Files may be gzip compressed. BED, `cache_tsv`, and `counts` use **0-based, half-open** coordinates. Standard Bismark `.cov` files use **1-based coordinates** and are converted to BED coordinates on input, following the [Bismark specification](https://felixkrueger.github.io/Bismark/usage/methylation-extraction/). Bismark `.zero.cov` files are detected automatically; use `--format bismark_cov_zero` for renamed zero-based files.

bedMethyl: coverage in column 10 and percent in column 11:

```text
chr1	100000	100001	.	0	+	100000	100001	0	4	75
```

Standard Bismark coverage: percent, methylated reads, unmethylated reads:

```text
chr1	100001	100001	75	3	1
```

Three-column cache (`--format cache_tsv --chrom chr1`): start, coverage, percent:

```text
100000	4	75
```

Counts (`--format counts`): `chrom start end methylated coverage`, or with `--chrom`, `start methylated coverage`. For example:

```text
chr1	100000	100001	3	4
```

Read counts must be integers with `0 <= methylated <= coverage`, and coverage must fit uint32. Percentages must be finite and between 0 and 100. Invalid rows fail with a filename and line number; they are not silently dropped or clipped. Duplicate positions retain the highest-coverage row with a warning. `track`, `browser`, and `#` header lines are ignored. Chromosome names must be valid path components because caches are stored by chromosome.

## Predictions and parameters

Defaults: ±25 kb window, 30 neighbors, covariance shrinkage 0.25, query noise floor 0.05, reference-library mean coverage strictly above 10, reference-site coverage at least 10. These parameters are configurable; use `--help` for all flags. Sites at query coverage 10 or above retain observed methylation by default. `--only-below` omits those rows. `--score-all` also computes prediction columns at deep sites; it cannot be combined with `--only-below`.

The default `pairwise` preset uses covariance estimated from jointly observed references and sends sites with panel SD below 5 percentage points to the panel mean. The `research` preset ranks neighbors by cosine similarity of standardized rows, fills missing reference values with column means before covariance estimation, and disables this SD cutoff. Both presets round absolute correlation ranking keys to 12 decimal places and break ties by distance, then genomic position, so numerical roundoff does not arbitrarily change equally correlated neighbors. Covariance uses the original values. `--sd-passthrough` changes the cutoff explicitly.

Fusion strength κ is calibrated using held-out reference reads, without fitting to the query. Calibration is recalculated when the selected references or fitting parameters change. `--kappa 8` overrides calibration; a two-column `depth kappa` table is also accepted. `--no-fuse` selects the context prediction. A coverage-zero site's fused prediction equals its context prediction.

`--method` accepts comma-separated additional columns: `panel_mean`, `neighbor_mean`, `local`, `global`, `comet`, `hmm`, `kalman`. BLUP is always included. Local and global require the `ml` extra. Additional methods can require considerably more time and memory. `--threads` controls CatBoost global fitting; numerical BLAS operations in CLI runs are limited to one thread for reproducibility. `--seed` controls random sampling.

The TSV includes original counts, observed percent, selected output percent (`pct`), context and fused estimates, panel mean and SD, neighbor count, and prediction source. `.gz` output uses ordinary gzip; no external compression executable is required. `<output>.run.json` records all effective arguments, package version, manifest hash, per-chromosome reference identities, κ, and run statistics.

## Docker

The container installs the package, includes all optional runtime dependencies pinned in `requirements-docker.lock`, and runs as UID/GID 10001. It requires no network at runtime. The base image receives upstream updates when rebuilt; refresh the dependency lock deliberately and run CI when updating it.

```bash
docker build -t wgbs-impute:0.2.1 .
mkdir -p demo_out
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$PWD/demo_out:/out" wgbs-impute:0.2.1 --demo
```

For a real run, create an **empty** writable output directory, mount references and query read-only, and preserve the query's identity or supply it explicitly:

```bash
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v /data/references:/refs:ro \
  -v /data/query.bed.gz:/query/query.bed.gz:ro \
  -v /data/out:/out \
  wgbs-impute:0.2.1 --preset research --sample-id SAMPLE
```

Container input/output paths are configurable with `--refs`, `--query`, and `--out`. Its default preset is `research`; the Python CLI defaults to `pairwise`. `--format`, `--chrom`, `--include-panel-sites`, `--max-cov`, `--only-below`, `--method`, `--n-refs`, `--disease-role`, `--sample-id`, fitting parameters, `--kappa`, `--threads`, `--seed`, and `--no-fuse` are available. The wrapper prepares a fresh panel each run. For repeated queries against one prepared panel, invoke the installed core CLI with `--entrypoint wgbs-impute` and mount the cache.

Outputs include `manifest.tsv`, `panel_cache/`, `imputed.tsv`, `imputed.tsv.run.json`, `imputed.bed.gz`, `missing_predictions.tsv`, `stats.tsv`, `stats_by_coverage.tsv`, `stats_by_sd.tsv`, and `dashboard.html`. **`run.complete.json` is written last**; its absence indicates an incomplete run. The output directory must be empty to prevent stale files from being mistaken for new results.

The BED contains only rows with a finite prediction or observed value. Rows with neither are retained in `imputed.tsv` and `missing_predictions.tsv`, and counted as `n_missing_predictions` in the statistics. They are never converted to 0% methylation. The BED retains original coverage: re-reading imputed coverage-zero rows as raw input does not create observed reads.

Statistics compare output values with available query observations; these differences are **not accuracy against ground truth**. The dashboard is a self-contained HTML file with no external network dependencies. Reference counts in the `all` row count unique libraries across chromosomes; per-chromosome counts can differ.

## Resource use and scope

Covariance matrices are fitted one site at a time rather than storing an `n_sites × neighbors × neighbors` tensor. Panel files are spooled to temporary arrays during preparation, and chromosomes are processed separately. Output rows and container summaries are streamed; exact SD medians retain 8 bytes per site per summary group. Panel beta/count arrays and neighbor indices still scale with sites × references and sites × neighbors. Parsing one input library and the query requires memory proportional to their input size. Optional methods may allocate additional dense arrays. Use chromosome-split inputs and `--chrom` for large workloads, and benchmark memory on your cohort before whole-genome deployment.

The synthetic demo checks software behavior and improvement over a panel-mean baseline using a controlled, recoverable signal: reference levels span 0.15–0.85, and the held-out query is at 0.15. It does not validate accuracy for every tissue, species, or assay. A real query close to the panel mean may not improve over that baseline. Validate an appropriate held-out cohort before relying on scientific results.

## Development and migration

```bash
python -m pip install '.[ml,dev]'
ruff check .
python -m unittest discover -s tests -v
python -m build
```

CI tests Python 3.10–3.13 installations, builds distribution artifacts, and runs the demo and regressions inside a non-root, read-only Docker container without runtime networking. See [RELEASE_NOTES.md](RELEASE_NOTES.md) for behavior changes and [PUBLISHING.md](PUBLISHING.md) for GitHub push instructions.

Version 0.2.1 uses cache schema 3. **Rebuild older caches, including 0.2.0 caches**, to regenerate neighbors with deterministic tie-breaking. Regenerate old Bismark-derived caches because their coordinates were previously interpreted incorrectly. Never reuse an old cache by copying or editing its metadata.
