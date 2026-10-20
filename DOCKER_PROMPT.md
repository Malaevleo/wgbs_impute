# Prompt

Build a Docker image for the `wgbs_impute` package in this repository. Python 3.11. Install `requirements.txt` (NumPy and CatBoost). Do not download methylation data into the image. Do not change the imputation math. The image only wraps the existing commands, writes the outputs below, and can run a small self-test.

## Run

Mount a reference directory and a query file. Write into a mounted output directory.

```bash
docker build -t wgbs-impute .
docker run --rm \
  -v /data/references:/refs:ro \
  -v /data/query.bed.gz:/query/query.bed.gz:ro \
  -v /data/out:/out \
  wgbs-impute \
  --preset research
```

`/refs` is a directory of `.bed` and `.bed.gz` files, one library per file. The same formats the package already reads are allowed: bedMethyl (coverage in column 10, percent in column 11), Bismark `.cov.gz`, and a three-column `start cov pct` cache when `--chrom` is set. An optional `/refs/samples.tsv` with columns `epirr`, `disease`, and `role` is the disease table. Libraries with `role=ref` are added; `role=target` is not, unless `--disease-role all`.

Inside the container the entrypoint runs, in order:

1. `python -m wgbs_impute make-manifest --panel-dir /refs --output /out/manifest.tsv`, plus `--disease-dir /refs` when `samples.tsv` is present.
2. `python -m wgbs_impute prepare-panel --manifest /out/manifest.tsv --cache /out/panel_cache --preset research` unless the user passed `--preset pairwise`.
3. `python -m wgbs_impute impute --query /query/query.bed.gz --panel-cache /out/panel_cache --output /out/imputed.tsv --preset research`. If the query filename stem matches a `sample_id` in the manifest, pass `--sample-id` so that library is not a reference for itself.

Flags after the image name are forwarded: `--preset`, `--max-cov`, `--only-below`, `--chrom`, `--n-refs`, `--disease-role`, `--method`. Default preset is `research`. Default method is fused BLUP. Do not turn on `local` or `global` unless the user names them.

## Outputs

All of these go in `/out`.

`imputed.bed.gz` is bedMethyl. Columns 1–3 and 6–9 are coordinates and placeholders. Column 10 is the original coverage. Column 11 is the imputed percent (`pct` from the tool). Add a header comment with the preset and κ. Keep the tool's `imputed.tsv` and `imputed.tsv.run.json` as well.

`stats.tsv` is one row per chromosome plus a `all` row. Columns:

- `chrom`, `n_sites`, `n_imputed`, `n_observed`, `n_panel_mean`, `n_passthrough`, `n_solved`
- `n_refs`, `kappa`
- `query_mean_cov` on sites with coverage above 0
- `frac_cov0`, `frac_cov_below_max`
- `mean_pct_imputed`, `mean_pct_observed` where the query has at least one read
- `mean_abs_imputed_minus_observed` on sites with coverage at least 1 and below `--max-cov` (this is not an error against truth; truth is not in the query file)
- `median_panel_sd`

`stats_by_coverage.tsv`: bins `0`, `1`, `2-4`, `5-9`, `10+`. For each bin, site count, mean imputed percent, and mean absolute difference between imputed and observed percent when observed exists.

`stats_by_sd.tsv`: panel-SD bins `0-5`, `5-10`, `10-15`, `15+`. Site count and fraction whose `source` is `panel_mean`.

## Dashboard

Write `/out/dashboard.html`, one file, no network, no extra server. Open it from the output directory. Four blocks:

1. Run summary: preset, reference count, κ per chromosome, sites written, sites solved, sites replaced by the panel mean.
2. Bar of `source` (`observed`, `blup_fused` or `blup`, `panel_mean`, `passthrough`).
3. The coverage-bin table as a bar of mean absolute imputed-minus-observed difference.
4. The SD-bin table as a bar of the fraction left on the panel mean.

Numbers on the page come only from `stats.tsv` and the two breakdown tables. Do not invent metrics.

## Smoke test

`docker run --rm -v "$PWD/demo_out":/out wgbs-impute --demo` does not read `/refs`. It builds a synthetic panel in a temp directory and checks that the image runs.

- 12 libraries, 400 CpGs on one chromosome, positions 100 bp apart, coverage about 40.
- Each library has one latent methylation level shared by sites 20 and up, so neighbors identify the library. The first 20 sites are identical in every library (panel SD 0).
- Hold out library `lib0`. Thin its counts to 2× with a hypergeometric draw from the full counts (the same rule as `audit_fix/thinning.py`: each depth is one draw from the full reads, methylated reads are a subset of the original methylated reads). Do not copy that file into the image if it is outside the repo; reimplement those few lines.
- Sites 0, 2, 4, … of the query get coverage 0. The odd sites keep the thinned reads.
- Run the real entrypoint on this panel and this query with `--preset research`.
- Score sites with original coverage at least 10 and query coverage below 10. Fused BLUP MAE against the held-out full-depth percent must be lower than the panel-mean MAE.
- Write the same BED, `stats.tsv`, breakdowns, and `dashboard.html` into `/out`.
- The demo is the test that the container works. `docker build` then `docker run --demo` must pass without a network and without any mounted reference data.

Do not put a real cohort in the image. Do not impute a whole genome in the demo.
