"""Container entrypoint. Wraps the imputation commands and writes a BED plus tables."""

from __future__ import annotations

import argparse
import csv
import html
import sys
from array import array
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from wgbs_impute.cli import main as tool_main
from wgbs_impute import __version__
from wgbs_impute.io import sample_id_from_filename
from wgbs_impute.impute import parse_kappa_arg
from wgbs_impute.thinning import subsample_depths
from wgbs_impute.validation import validate_options

COVERAGE_BINS = ("0", "1", "2-4", "5-9", "10+")
SD_BINS = ("0-5", "5-10", "10-15", "15+")


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wgbs-impute")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--refs", type=Path, default=Path("/refs"))
    parser.add_argument("--query", type=Path, default=Path("/query/query.bed.gz"))
    parser.add_argument("--out", type=Path, default=Path("/out"))
    parser.add_argument("--preset", choices=("research", "pairwise"), default="research")
    parser.add_argument("--max-cov", type=float, default=10.0)
    parser.add_argument("--only-below", action="store_true")
    parser.add_argument("--include-panel-sites", action="store_true")
    parser.add_argument("--chrom", default=None)
    parser.add_argument("--n-refs", type=int, default=0)
    parser.add_argument("--disease-role", default="ref", choices=("ref", "target", "all"))
    parser.add_argument("--method", default="blup")
    parser.add_argument(
        "--sample-id", help="reference sample identity to exclude; inferred from query name by default"
    )
    parser.add_argument(
        "--format",
        default="auto",
        choices=("auto", "bedmethyl", "bismark_cov", "bismark_cov_zero", "cache_tsv", "counts"),
    )
    parser.add_argument("--window", type=int, default=25000)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--shrink", type=float, default=0.25)
    parser.add_argument("--noise-floor", type=float, default=0.05)
    parser.add_argument("--sd-passthrough", type=float)
    parser.add_argument("--neighbor-min-cov", type=int, default=1)
    parser.add_argument("--min-ref-mean-cov", type=float, default=10)
    parser.add_argument("--min-ref-site-cov", type=int, default=10)
    parser.add_argument("--kappa")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fuse", action="store_true")
    args = parser.parse_args(argv)
    validate_options(**{k: v for k, v in vars(args).items() if v is not None})
    if args.demo:
        return run_demo(args.out)
    if not args.refs.is_dir():
        raise SystemExit(f"reference directory not found: {args.refs}")
    if not args.query.is_file():
        raise SystemExit(f"query file not found: {args.query}")
    run_pipeline(
        args.refs,
        args.query,
        args.out,
        preset=args.preset,
        max_cov=args.max_cov,
        only_below=args.only_below,
        chrom=args.chrom,
        n_refs=args.n_refs,
        disease_role=args.disease_role,
        method=args.method,
        sample_id=args.sample_id,
        fmt=args.format,
        options={
            k: getattr(args, k)
            for k in (
                "window",
                "neighbors",
                "shrink",
                "noise_floor",
                "sd_passthrough",
                "neighbor_min_cov",
                "min_ref_mean_cov",
                "min_ref_site_cov",
                "kappa",
                "threads",
                "seed",
                "no_fuse",
                "include_panel_sites",
            )
        },
    )
    return 0


def run_pipeline(
    refs: Path,
    query: Path,
    out: Path,
    *,
    preset: str,
    max_cov: float,
    only_below: bool,
    chrom: str | None,
    n_refs: int,
    disease_role: str,
    method: str,
    sample_id: str | None = None,
    fmt: str = "auto",
    options: dict | None = None,
) -> None:
    options = options or {}
    validate_options(**{k: v for k, v in options.items() if v is not None})
    if options.get("kappa") is not None:
        parse_kappa_arg(options["kappa"])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"output directory must be empty: {out}; choose a new directory")
    manifest = out / "manifest.tsv"
    make = [
        "make-manifest",
        "--panel-dir",
        str(refs),
        "--output",
        str(manifest),
        "--disease-role",
        disease_role,
    ]
    if (refs / "samples.tsv").exists():
        make.extend(["--disease-dir", str(refs)])
    if tool_main(make) != 0:
        raise SystemExit("make-manifest failed")
    cache = out / "panel_cache"
    shared = []
    for key in (
        "window",
        "neighbors",
        "shrink",
        "sd_passthrough",
        "min_ref_mean_cov",
        "min_ref_site_cov",
        "seed",
    ):
        if options.get(key) is not None:
            shared.extend(["--" + key.replace("_", "-"), str(options[key])])
    if (
        tool_main(
            ["prepare-panel", "--manifest", str(manifest), "--cache", str(cache), "--preset", preset, *shared]
        )
        != 0
    ):
        raise SystemExit("prepare-panel failed")
    impute = [
        "impute",
        "--query",
        str(query),
        "--panel-cache",
        str(cache),
        "--output",
        str(out / "imputed.tsv"),
        "--preset",
        preset,
        "--max-cov",
        str(max_cov),
        "--n-refs",
        str(n_refs),
        "--method",
        method,
    ]
    impute.extend(shared)
    impute.extend(["--format", fmt])
    for key in ("noise_floor", "neighbor_min_cov", "kappa", "threads"):
        if options.get(key) is not None:
            impute.extend(["--" + key.replace("_", "-"), str(options[key])])
    if options.get("no_fuse"):
        impute.append("--no-fuse")
    if options.get("include_panel_sites"):
        impute.append("--include-panel-sites")
    if only_below:
        impute.append("--only-below")
    if chrom:
        impute.extend(["--chrom", chrom])
    sid = sample_id if sample_id is not None else sample_id_from_filename(query)
    ids = _manifest_ids(manifest)
    if sid in ids or sample_id is not None:
        impute.extend(["--sample-id", sid])
    if tool_main(impute) != 0:
        raise SystemExit("impute failed")
    result_path = out / "imputed.tsv"
    report = json.loads((out / "imputed.tsv.run.json").read_text())
    write_bed(out / "imputed.bed.gz", _read_tsv(result_path), preset, report)
    with result_path.open() as source, (out / "missing_predictions.tsv").open("w", newline="") as dest:
        reader = csv.DictReader(source, delimiter="\t")
        writer = csv.DictWriter(dest, fieldnames=reader.fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in reader:
            if not row.get("pct"):
                writer.writerow(row)
    stats, by_cov, by_sd = summarize(_read_tsv(result_path), report, max_cov)
    _write_tsv(out / "stats.tsv", stats)
    _write_tsv(out / "stats_by_coverage.tsv", by_cov)
    _write_tsv(out / "stats_by_sd.tsv", by_sd)
    (out / "dashboard.html").write_text(render_dashboard(stats, by_cov, by_sd, preset))
    (out / "run.complete.json").write_text(json.dumps({"version": report["version"], "status": "complete"}))


def write_bed(path: Path, rows: list[dict], preset: str, report: dict) -> None:
    kappas = []
    for chrom, info in report.get("chromosomes", {}).items():
        if "kappa" in info:
            kappas.append(f"{chrom}={float(info['kappa']):.4f}")
    header = f"# wgbs_impute preset={preset} kappa={','.join(kappas)}\n"
    with gzip.open(path, "wt") as fh:
        fh.write(header)
        for row in rows:
            pct = row.get("pct")
            if pct is None or pct == "":
                continue
            fh.write(
                f"{row['chrom']}\t{row['start']}\t{row['end']}\t.\t0\t.\t"
                f"{row['start']}\t{row['end']}\t0,0,0\t{row['cov']}\t{pct}\n"
            )


class _Accumulator:
    def __init__(self):
        self.n = 0
        self.sources = Counter()
        self.sums = Counter()
        self.counts = Counter()
        self.sd = array("d")
        self.coverage = {name: Counter() for name in COVERAGE_BINS}
        self.sd_bins = {name: Counter() for name in SD_BINS}

    def add(self, row, max_cov):
        self.n += 1
        self.sources[row["source"]] += 1
        cov = float(row["cov"])
        pct = float(row["pct"]) if row.get("pct") else float("nan")
        obs = float(row["pct_obs"]) if row.get("pct_obs") else float("nan")
        sd = float(row["panel_sd"]) if row.get("panel_sd") else float("nan")
        self.counts["zero"] += cov == 0
        self.counts["low"] += cov < max_cov
        self.counts["missing"] += not np.isfinite(pct)
        for name, value in (("cov", cov if cov > 0 else float("nan")), ("pct", pct), ("obs", obs)):
            if np.isfinite(value):
                self.sums[name] += value
                self.counts[name] += 1
        diff = abs(pct - obs)
        if 1 <= cov < max_cov and np.isfinite(diff):
            self.sums["diff"] += diff
            self.counts["diff"] += 1
        name = "0" if cov == 0 else "1" if cov == 1 else "2-4" if cov < 5 else "5-9" if cov < 10 else "10+"
        bucket = self.coverage[name]
        bucket["n"] += 1
        if np.isfinite(pct):
            bucket["pct"] += pct
            bucket["n_pct"] += 1
        if cov >= 1 and np.isfinite(diff):
            bucket["diff"] += diff
            bucket["n_diff"] += 1
        if np.isfinite(sd):
            self.sd.append(sd)
            name = "0-5" if sd < 5 else "5-10" if sd < 10 else "10-15" if sd < 15 else "15+"
            self.sd_bins[name]["n"] += 1
            self.sd_bins[name]["mean"] += row["source"] == "panel_mean"

    def stats(self, chrom, report):
        info = report.get("chromosomes", {}).get(chrom, {})
        if chrom == "all":
            infos = list(report.get("chromosomes", {}).values())
            kappa = float(np.mean([i["kappa"] for i in infos])) if infos else float("nan")
            solved = sum(i.get("n_solved", 0) for i in infos)
            refs = len(report.get("panel_sample_ids", []))
        else:
            kappa, solved, refs = (
                info.get("kappa", float("nan")),
                info.get("n_solved", 0),
                info.get("n_libraries", 0),
            )
        return {
            "chrom": chrom,
            "n_sites": self.n,
            "n_imputed": self.sources["blup"] + self.sources["blup_fused"],
            "n_observed": self.sources["observed"],
            "n_panel_mean": self.sources["panel_mean"],
            "n_passthrough": self.sources["passthrough"],
            "n_missing_predictions": self.counts["missing"],
            "n_solved": solved,
            "n_refs": refs,
            "kappa": _num(kappa),
            "query_mean_cov": _ratio(self.sums["cov"], self.counts["cov"]),
            "frac_cov0": _ratio(self.counts["zero"], self.n),
            "frac_cov_below_max": _ratio(self.counts["low"], self.n),
            "mean_pct_imputed": _ratio(self.sums["pct"], self.counts["pct"]),
            "mean_pct_observed": _ratio(self.sums["obs"], self.counts["obs"]),
            "mean_abs_imputed_minus_observed": _ratio(self.sums["diff"], self.counts["diff"]),
            "median_panel_sd": _num(float(np.median(self.sd)) if self.sd else float("nan")),
        }


def _ratio(total, count):
    return _num(total / count) if count else ""


def summarize(rows, report: dict, max_cov: float):
    """Stream rows; retain only panel SDs for an exact median (8 bytes/site/group)."""
    overall = _Accumulator()
    groups = {}
    for row in rows:
        if row["chrom"] not in groups:
            groups[row["chrom"]] = _Accumulator()
        groups[row["chrom"]].add(row, max_cov)
        overall.add(row, max_cov)
    named = list(groups.items()) + [("all", overall)]
    stats = [group.stats(chrom, report) for chrom, group in named]
    by_cov, by_sd = [], []
    for chrom, group in named:
        for name, values in group.coverage.items():
            by_cov.append(
                {
                    "chrom": chrom,
                    "bin": name,
                    "n_sites": values["n"],
                    "mean_pct_imputed": _ratio(values["pct"], values["n_pct"]),
                    "mean_abs_imputed_minus_observed": _ratio(values["diff"], values["n_diff"]),
                }
            )
        for name, values in group.sd_bins.items():
            by_sd.append(
                {
                    "chrom": chrom,
                    "bin": name,
                    "n_sites": values["n"],
                    "frac_panel_mean": _ratio(values["mean"], values["n"]),
                }
            )
    return stats, by_cov, by_sd


def render_dashboard(stats, by_cov, by_sd, preset: str) -> str:
    overall = next(row for row in stats if row["chrom"] == "all")
    kappas = ", ".join(
        f"{html.escape(str(row['chrom']))} {row['kappa']}" for row in stats if row["chrom"] != "all"
    )
    sources = [
        ("observed", overall["n_observed"]),
        ("imputed", overall["n_imputed"]),
        ("panel mean", overall["n_panel_mean"]),
        ("passthrough", overall["n_passthrough"]),
    ]
    cov_rows = [r for r in by_cov if r["chrom"] == "all"]
    sd_rows = [r for r in by_sd if r["chrom"] == "all"]
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>wgbs_impute</title>
<style>
body {{ font: 16px/1.4 sans-serif; margin: 32px; color: #111; background: #fff; }}
h1 {{ font-size: 22px; }} h2 {{ font-size: 16px; margin-top: 28px; }}
.row {{ display: grid; grid-template-columns: 110px 1fr 72px; gap: 8px; align-items: center; margin: 4px 0; }}
.bar {{ background: #2370fa; height: 14px; }}
.note {{ color: #595f74; }}
</style></head><body>
<h1>wgbs_impute</h1>
<p class="note">Preset {preset}. References {overall["n_refs"]}. κ {kappas}.</p>
<p>Sites written {overall["n_sites"]}. Solved {overall["n_solved"]}.
Panel mean {overall["n_panel_mean"]}.</p>
<h2>Source</h2>
{_bars(sources)}
<h2>Imputed minus observed, by query coverage</h2>
{_bars([(r["bin"], _f(r["mean_abs_imputed_minus_observed"])) for r in cov_rows])}
<h2>Fraction left on the panel mean, by panel SD</h2>
{_bars([(r["bin"], _f(r["frac_panel_mean"]) * 100) for r in sd_rows], percent=True)}
</body></html>
"""


def _bars(pairs, percent=False) -> str:
    vals = [float(v) for _, v in pairs]
    top = 100.0 if percent else max(vals) if vals and max(vals) > 0 else 1.0
    lines = []
    for label, value in pairs:
        width = 100.0 * float(value) / top
        suffix = f"{float(value):.1f}%" if percent else f"{float(value):.2f}"
        lines.append(
            f'<div class="row"><span>{html.escape(str(label))}</span>'
            f'<div class="bar" style="width:{width:.1f}%"></div><span>{suffix}</span></div>'
        )
    return "\n".join(lines)


def run_demo(out: Path) -> int:
    import tempfile

    n_sites, n_libs = 400, 12
    pos = np.arange(n_sites) * 100
    # A known, recoverable signal: held-out lib0 differs substantially from
    # the remaining panel. A random query near the panel mean cannot reliably
    # beat that baseline at 2x coverage, even when the implementation is correct.
    z = np.linspace(0.15, 0.85, n_libs)
    beta = np.tile(z[:, None], (1, n_sites))
    beta[:, :20] = 0.2
    counts = np.full((n_libs, n_sites), 80, dtype=np.int64)
    meth = np.rint(counts * beta).astype(np.int64)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        refs = tmp / "refs"
        refs.mkdir()
        for i in range(n_libs):
            _write_bed(refs / f"lib{i}.bed.gz", "chr1", pos, meth[i], counts[i])
        mean = float(counts[0].mean())
        thin_m, thin_c = _thin_hypergeometric(meth[0], counts[0], 2.0, mean, 7)
        q_m = thin_m.copy()
        q_c = thin_c.copy()
        q_c[0::2] = 0
        q_m[0::2] = 0
        query = tmp / "lib0.bed.gz"
        _write_bed(query, "chr1", pos, q_m, q_c)
        run_pipeline(
            refs,
            query,
            out,
            preset="research",
            max_cov=10,
            only_below=False,
            chrom=None,
            n_refs=0,
            disease_role="ref",
            method="blup",
        )
        table = {int(r["start"]): r for r in _read_tsv(out / "imputed.tsv")}
        err_f, err_p = [], []
        truth = 100.0 * meth[0] / counts[0]
        for s, start in enumerate(pos):
            if counts[0, s] < 10 or q_c[s] >= 10:
                continue
            row = table[int(start)]
            if not row["pct"] or not row["pct_panel"]:
                continue
            err_f.append(abs(float(row["pct"]) - truth[s]))
            err_p.append(abs(float(row["pct_panel"]) - truth[s]))
        if not err_f or float(np.mean(err_f)) >= float(np.mean(err_p)):
            raise SystemExit(
                f"demo failed: fused MAE {np.mean(err_f) if err_f else float('nan')} "
                f"panel {np.mean(err_p) if err_p else float('nan')}"
            )
        print(f"demo ok: fused MAE {np.mean(err_f):.3f}  panel MAE {np.mean(err_p):.3f}", flush=True)
    return 0


def _thin_hypergeometric(meth, cov, depth, mean_full, seed):
    return subsample_depths(meth, cov, [depth], mean_full, seed)[depth]


def _write_bed(path: Path, chrom, pos, meth, cov) -> None:
    lines = []
    for s, m, c in zip(pos, meth, cov):
        pct = 0.0 if c <= 0 else 100.0 * float(m) / float(c)
        start, end = int(s), int(s) + 1
        lines.append(f"{chrom}\t{start}\t{end}\t.\t0\t.\t{start}\t{end}\t0,0,0\t{int(c)}\t{pct:.4f}\n")
    with gzip.open(path, "wt") as fh:
        fh.writelines(lines)


def _read_tsv(path: Path) -> list[dict]:
    with path.open() as fh:
        yield from csv.DictReader(fh, delimiter="\t")


def _write_tsv(path: Path, rows: list[dict], header=None) -> None:
    header = header or list(rows[0])
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _manifest_ids(path: Path) -> set[str]:
    rows = _read_tsv(path)
    return {row["sample_id"] for row in rows}


def _num(value) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.6g}"


def _f(text) -> float:
    return float(text) if text else 0.0


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except (OSError, ValueError) as exc:
        print(f"wgbs-impute: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
