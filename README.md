# wgbs_impute

Impute methylation at CpG sites that do not have enough reads. The default is fused BLUP: a per-site prediction from up to 30 neighboring CpGs within 25 kb, mixed with the query's own reads. The panel is the manifest you pass. A site on one chromosome never takes a neighbor from another.

## Installation

Python 3.9 or newer.

```bash
git clone https://github.com/Malaevleo/wgbs_impute.git
cd wgbs_impute
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The clone directory is the package, so run commands from its parent, or set `PYTHONPATH` to that parent:

```bash
export PYTHONPATH="$(pwd)/.."
python -m wgbs_impute --help
```

`requirements.txt` installs NumPy and CatBoost, including the methods `local` and `global`. Comet, the copying HMM, and the Kalman smoother use NumPy only. A path ending in `.gz` is written with `bgzip` when that program is installed, and with ordinary gzip otherwise.

## Docker

The image runs the same three commands. Mount the reference BED directory, the query file, and an output directory.

```bash
docker build -t wgbs-impute .
docker run --rm \
  -v /data/references:/refs:ro \
  -v /data/query.bed.gz:/query/query.bed.gz:ro \
  -v /data/out:/out \
  wgbs-impute --preset research
```

`/out` receives `imputed.bed.gz`, `stats.tsv`, `stats_by_coverage.tsv`, `stats_by_sd.tsv`, and `dashboard.html`. `docker run --rm -v "$PWD/demo_out":/out wgbs-impute --demo` builds a synthetic panel inside the container and checks that imputation beats the panel mean.

## Quickstart

Put the reference BED files in one directory. `make-manifest` writes a table that points at each file. `prepare-panel` reads those files once and stores the panel. `impute` uses that store on a low-coverage query.

```bash
python -m wgbs_impute make-manifest --panel-dir references/ --output manifest.tsv
python -m wgbs_impute prepare-panel --manifest manifest.tsv --cache panel_cache/
python -m wgbs_impute impute --query query.bed.gz --panel-cache panel_cache/ --output query.imputed.tsv
```

`manifest.tsv` is that table, one reference library per row. It is not a copy of the methylation values. The columns are `path`, `sample_id`, `group`, and `chrom`. Only `path` and `sample_id` are required. `group` is a label used when you ask for a smaller panel, and it is printed in the log. `chrom` is set when the filename contains it, as in `SAMPLE_CellType.chr1.bed.gz`. A multi-chromosome BED leaves it blank. `mean_cov` is optional. If you omit it, `prepare-panel` measures mean coverage from the file and drops libraries at or below 10×.

`make-manifest` looks in that directory only, not in subfolders, for `.bed` and `.bed.gz`. The sample id is the part of the filename before the first `_` or `.`, so `SAMPLE01_CellType.chr1.bed.gz` becomes `SAMPLE01`. If the query's `sample_id` is in the manifest, it is removed from the references at impute time.

Disease libraries stay in their own folder. Pass that folder to add them to the same manifest. Their `group` column is the disease name. If `samples.tsv` sits in that folder or next to it, with columns `epirr`, `disease`, and `role`, only rows with `role` `ref` are added. `--disease-role all` also adds the libraries marked `target`.

```bash
python -m wgbs_impute make-manifest --panel-dir references/ --disease-dir disease_libraries/ --output manifest.tsv
```

`--preset research` ranks neighbors by a cosine of z-scored rows and fills missing reference values with the column mean before the covariance. It also solves stable sites instead of replacing them with the panel mean. The default `--preset pairwise` keeps a pair only when both references observed it.

By default, sites with coverage below 10 are fused BLUP and sites at 10 or above keep the observed percent. `--only-below` writes just the low-coverage rows. `--no-fuse` writes the context prediction. `--method` adds other columns: `panel_mean`, `neighbor_mean`, `local`, `global`, `comet`, `hmm`, `kalman`. Local and global need CatBoost and are off unless named.

## Inputs

Gzip is fine. Coordinates are 0-based start. Coverage is not capped at 255. A site with coverage 0 is imputed, not dropped.

bedMethyl, coverage in column 10 and percent in column 11:

```
chr1	100000	100001	.	0	+	100000	100001	0	4	75
```

Bismark `.cov` / `.cov.gz`, percent, methylated count, unmethylated count:

```
chr1	100000	100001	75	3	1
```

Three-column cache (`--format cache_tsv --chrom chr1`): start, coverage, percent.

```
100000	4	75
```

Counts (`--format counts`): `chrom start end methylated coverage`, or with `--chrom` the three columns `start methylated coverage`.

## Defaults

Window 25 kb, 30 neighbors, shrink 0.25, noise floor 0.05, full manifest. `--window` changes the neighbor search. `--n-refs` keeps the deepest libraries, balanced across `group`. Sites whose reference SD is below 5 percentage points get the panel mean and skip the solve, unless you are on `--preset research` or you set `--sd-passthrough 0`.

The fusion weight κ is estimated while preparing the panel, by thinning the deepest library, and is not fit on the query. `--kappa` overrides it.

The output is a headered TSV. `<output>.run.json` records the arguments, the reference ids actually used, and the κ of each chromosome.
