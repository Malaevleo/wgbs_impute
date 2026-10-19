# wgbs_impute

Impute methylation at CpG sites that do not have enough reads. The default is fused BLUP: a per-site prediction from up to 30 neighboring CpGs within 25 kb, mixed with the query's own reads. The panel is the manifest you pass. A site on one chromosome never takes a neighbor from another.

```
python -m wgbs_impute prepare-panel --manifest panel.tsv --cache panel_cache/
python -m wgbs_impute impute --query query.bed.gz --panel-cache panel_cache/ --output query.imputed.tsv
```

`panel.tsv` has a header and one library per row: `path`, `sample_id`, and optional `group`, `chrom`, `mean_cov`. Libraries at or below 10× mean coverage are dropped. If the query's `sample_id` is in the manifest, it is removed from the references at impute time.

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
