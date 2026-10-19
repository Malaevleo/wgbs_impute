# wgbs_impute

Impute methylation at CpG sites that do not have enough reads. The default is fused BLUP: a per-site prediction from up to 30 neighboring CpGs within 25 kb, mixed with the query's own reads. The panel is the manifest you pass. A site on one chromosome never takes a neighbor from another.

## Quickstart

Put the reference BED files in one directory. `make-manifest` writes a table that points at each file. `prepare-panel` reads those files once and stores the panel. `impute` uses that store on a low-coverage query.

```
python -m wgbs_impute make-manifest --panel-dir references/ --output manifest.tsv
python -m wgbs_impute prepare-panel --manifest manifest.tsv --cache panel_cache/
python -m wgbs_impute impute --query query.bed.gz --panel-cache panel_cache/ --output query.imputed.tsv
```

`manifest.tsv` is that table, one reference library per row. It is not a copy of the methylation values. The columns are `path`, `sample_id`, `group`, and `chrom`. Only `path` and `sample_id` are required. `group` is a lineage or disease label, used when you ask for a smaller panel and printed in the log. `chrom` is set when the filename contains it, as in `sample.chr21.bed.gz`; a multi-chromosome BED leaves it blank. `mean_cov` is optional. If you omit it, `prepare-panel` measures mean coverage from the file and drops libraries at or below 10×.

`make-manifest` looks in that directory only, not in subfolders, for `.bed` and `.bed.gz`. The sample id is the part of the filename before the first `_` or `.`, so `IHECRE00000155_B_Lymphocyte.chr21.bed.gz` becomes `IHECRE00000155`. Fill in `group` yourself if you want lineage labels. If the query's `sample_id` is in the manifest, it is removed from the references at impute time.

Disease libraries stay in their own folder. Pass that folder to add them to the same manifest. Their `group` column is the disease name. If `samples.tsv` sits in that folder or next to it, with columns `epirr`, `disease`, and `role`, only rows with `role` `ref` are added. `--disease-role all` also adds the libraries marked `target`.

```
python -m wgbs_impute make-manifest --panel-dir references/ --disease-dir disease_libraries/ --output manifest.tsv
```

`--preset research` uses the BLUP from the chromosome-21 audits: neighbors ranked by a cosine of z-scored rows, and missing reference values filled with the column mean before the covariance. That is the estimator behind the published scores. The default `--preset pairwise` keeps a pair only when both references observed it. Research mode also solves stable sites instead of replacing them with the panel mean.

By default, sites with coverage below 10 are fused BLUP and sites at 10 or above keep the observed percent. `--only-below` writes just the low-coverage rows. `--no-fuse` writes the context prediction. `--method` adds other columns: `panel_mean`, `neighbor_mean`, `local`, `global`, `comet`, `hmm`, `kalman`. Local and global need CatBoost and are off unless named.

## Inputs

Gzip is fine. Coordinates are GRCh38, 0-based start. Coverage is not capped at 255. A site with coverage 0 is imputed, not dropped.

bedMethyl, coverage in column 10 and percent in column 11:

```
chr21	43823910	43823911	.	0	+	43823910	43823911	0	4	75
```

Bismark `.cov` / `.cov.gz`, percent, methylated count, unmethylated count:

```
chr21	43823910	43823911	75	3	1
```

Three-column cache (`--format cache_tsv --chrom chr21`): start, coverage, percent.

```
43823910	4	75
```

Counts (`--format counts`): `chrom start end methylated coverage`, or with `--chrom` the three columns `start methylated coverage`.

## Defaults

Window 25 kb, 30 neighbors, shrink 0.25, noise floor 0.05, full manifest. `--window 100000` is available; it helped on one chromosome-21 eval set and is not the default. `--n-refs` keeps the deepest libraries, balanced across `group`, and prints a warning: a panel of 50 failed for myeloid targets. Sites whose reference SD is below 5 percentage points get the panel mean and skip the solve.

The fusion weight κ is estimated while preparing the panel, by thinning the deepest library, and is not fit on the query. `--kappa` overrides it.

The output is a headered TSV. A path ending in `.gz` is bgzipped when `bgzip` is installed, otherwise gzipped. `<output>.run.json` records the arguments, the reference ids actually used, and the κ of each chromosome.

## Limits

The measurements behind these defaults are blood, mostly chromosome 21, made by thinning deep libraries rather than by sequencing real low-pass libraries. Myeloid targets wanted the full panel. Disease libraries in a healthy panel cost accuracy, so they belong in the manifest only when the query is the same disease. Stable CpGs are the panel mean. The numbers are in `audit_opus/SUMMARY.md`.
