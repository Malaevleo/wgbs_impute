"""Command line for panel preparation and imputation."""

from __future__ import annotations

import argparse
import sys

from threadpoolctl import threadpool_limits
from pathlib import Path

from wgbs_impute.impute import impute_query, parse_kappa_arg
from wgbs_impute import __version__
from wgbs_impute.io import FORMATS, write_manifest
from wgbs_impute.validation import validate_options
from wgbs_impute.panel import prepare_panel


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wgbs_impute")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    make = sub.add_parser("make-manifest", help="list BED files in a directory as a panel manifest")
    make.add_argument("--panel-dir", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument(
        "--disease-dir", type=Path, default=None, help="extra folder of disease libraries to add to the panel"
    )
    make.add_argument(
        "--disease-table",
        type=Path,
        default=None,
        help="TSV with epirr, disease, and optional role; default samples.tsv next to --disease-dir",
    )
    make.add_argument(
        "--disease-role",
        default="ref",
        choices=("ref", "target", "all"),
        help="which rows of the disease table to add (default: ref)",
    )

    prep = sub.add_parser("prepare-panel", help="cache a reference panel, one chromosome at a time")
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--cache", type=Path, required=True)
    prep.add_argument("--min-ref-mean-cov", type=float, default=10.0)
    prep.add_argument("--min-ref-site-cov", type=int, default=10)
    prep.add_argument("--window", type=int, default=25000)
    prep.add_argument("--neighbors", type=int, default=30)
    prep.add_argument("--shrink", type=float, default=0.25)
    prep.add_argument("--sd-passthrough", type=float, default=None)
    prep.add_argument("--preset", choices=("pairwise", "research"), default="pairwise")
    prep.add_argument("--seed", type=int, default=42)

    imp = sub.add_parser("impute", help="impute low-coverage CpGs with fused BLUP")
    imp.add_argument("--query", type=Path, required=True)
    imp.add_argument("--output", type=Path, required=True)
    imp.add_argument("--panel-cache", type=Path)
    imp.add_argument("--panel-manifest", type=Path)
    imp.add_argument("--format", default="auto", choices=("auto",) + FORMATS)
    imp.add_argument("--chrom", default=None)
    imp.add_argument("--regions", type=Path)
    imp.add_argument("--max-cov", type=float, default=10.0)
    imp.add_argument("--only-below", action="store_true")
    imp.add_argument(
        "--include-panel-sites",
        action="store_true",
        help="add absent reference-panel sites as zero-coverage query rows",
    )
    imp.add_argument("--score-all", action="store_true")
    imp.add_argument("--sample-id", default=None)
    imp.add_argument("--min-ref-mean-cov", type=float, default=10.0)
    imp.add_argument("--min-ref-site-cov", type=int, default=10)
    imp.add_argument("--n-refs", type=int, default=0)
    imp.add_argument("--window", type=int, default=25000)
    imp.add_argument("--neighbors", type=int, default=30)
    imp.add_argument("--shrink", type=float, default=0.25)
    imp.add_argument("--noise-floor", type=float, default=0.05)
    imp.add_argument("--neighbor-min-cov", type=int, default=1)
    imp.add_argument("--sd-passthrough", type=float, default=None)
    imp.add_argument("--preset", choices=("pairwise", "research"), default="pairwise")
    imp.add_argument("--fuse", dest="fuse_on", action="store_true", default=True)
    imp.add_argument("--no-fuse", dest="fuse_on", action="store_false")
    imp.add_argument("--kappa", default=None)
    imp.add_argument("--threads", type=int, default=1)
    imp.add_argument("--seed", type=int, default=42)
    imp.add_argument("--method", default="blup", help="comma-separated; default blup")
    imp.add_argument("--local-iterations", type=int, default=120)
    imp.add_argument("--local-depth", type=int, default=4)
    imp.add_argument("--max-train", type=int, default=20000)
    imp.add_argument("--refnn-k", type=int, default=5)

    args = parser.parse_args(argv)
    if getattr(args, "sd_passthrough", None) is None and args.cmd != "make-manifest":
        args.sd_passthrough = 0.0 if getattr(args, "preset", "pairwise") == "research" else 5.0
    validate_options(**vars(args))
    if args.cmd == "make-manifest":
        rows = write_manifest(
            args.panel_dir,
            args.output,
            disease_dir=args.disease_dir,
            disease_table=args.disease_table,
            disease_role=args.disease_role,
        )
        print(f"wrote {len(rows)} libraries to {args.output}", flush=True)
        return 0
    if args.cmd == "prepare-panel":
        prepare_panel(
            args.manifest,
            args.cache,
            min_ref_mean_cov=args.min_ref_mean_cov,
            min_ref_site_cov=args.min_ref_site_cov,
            window=args.window,
            neighbors=args.neighbors,
            shrink=args.shrink,
            sd_passthrough=args.sd_passthrough,
            seed=args.seed,
            preset=args.preset,
        )
        return 0
    if not args.panel_cache and not args.panel_manifest:
        parser.error("impute needs --panel-cache or --panel-manifest")
    impute_query(
        args.query,
        args.output,
        panel_cache=args.panel_cache,
        panel_manifest=args.panel_manifest,
        fmt=args.format,
        chrom=args.chrom,
        regions=args.regions,
        max_cov=args.max_cov,
        only_below=args.only_below,
        score_all=args.score_all,
        sample_id=args.sample_id,
        min_ref_mean_cov=args.min_ref_mean_cov,
        min_ref_site_cov=args.min_ref_site_cov,
        n_refs=args.n_refs,
        window=args.window,
        neighbors=args.neighbors,
        shrink=args.shrink,
        noise_floor=args.noise_floor,
        neighbor_min_cov=args.neighbor_min_cov,
        sd_passthrough=args.sd_passthrough,
        fuse_on=args.fuse_on,
        kappa=parse_kappa_arg(args.kappa),
        threads=args.threads,
        seed=args.seed,
        methods=[m.strip() for m in args.method.split(",") if m.strip()],
        local_iterations=args.local_iterations,
        local_depth=args.local_depth,
        max_train=args.max_train,
        refnn_k=args.refnn_k,
        preset=args.preset,
        include_panel_sites=args.include_panel_sites,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        # Bound numerical library thread pools; CatBoost also receives --threads.
        # Prepare-panel is deliberately single-threaded for reproducibility.
        with threadpool_limits(limits=1):
            return _main(argv)
    except (OSError, ValueError) as exc:
        print(f"wgbs_impute: {exc}", file=sys.stderr)
        return 2
