"""Container entrypoint. Wraps the imputation commands and writes a BED plus tables."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from wgbs_impute.cli import main as tool_main
from wgbs_impute.io import sample_id_from_filename

COVERAGE_BINS = ("0", "1", "2-4", "5-9", "10+")
SD_BINS = ("0-5", "5-10", "10-15", "15+")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wgbs-impute")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--refs", type=Path, default=Path("/refs"))
    parser.add_argument("--query", type=Path, default=Path("/query/query.bed.gz"))
    parser.add_argument("--out", type=Path, default=Path("/out"))
    parser.add_argument("--preset", choices=("research", "pairwise"), default="research")
    parser.add_argument("--max-cov", type=float, default=10.0)
    parser.add_argument("--only-below", action="store_true")
    parser.add_argument("--chrom", default=None)
    parser.add_argument("--n-refs", type=int, default=0)
    parser.add_argument("--disease-role", default="ref", choices=("ref", "target", "all"))
    parser.add_argument("--method", default="blup")
    args = parser.parse_args(argv)
    if args.demo:
        return run_demo(args.out)
    if not args.refs.is_dir():
        raise SystemExit(f"reference directory not found: {args.refs}")
    if not args.query.is_file():
        raise SystemExit(f"query file not found: {args.query}")
    run_pipeline(
        args.refs, args.query, args.out,
        preset=args.preset, max_cov=args.max_cov, only_below=args.only_below,
        chrom=args.chrom, n_refs=args.n_refs, disease_role=args.disease_role,
        method=args.method,
    )
    return 0


def run_pipeline(
    refs: Path, query: Path, out: Path, *,
    preset: str, max_cov: float, only_below: bool, chrom: str | None,
    n_refs: int, disease_role: str, method: str,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.tsv"
    make = ["make-manifest", "--panel-dir", str(refs), "--output", str(manifest),
            "--disease-role", disease_role]
    if (refs / "samples.tsv").exists():
        make.extend(["--disease-dir", str(refs)])
    if tool_main(make) != 0:
        raise SystemExit("make-manifest failed")
    cache = out / "panel_cache"
    if tool_main(["prepare-panel", "--manifest", str(manifest), "--cache", str(cache),
                  "--preset", preset]) != 0:
        raise SystemExit("prepare-panel failed")
    impute = ["impute", "--query", str(query), "--panel-cache", str(cache),
              "--output", str(out / "imputed.tsv"), "--preset", preset,
              "--max-cov", str(max_cov), "--n-refs", str(n_refs), "--method", method]
    if only_below:
        impute.append("--only-below")
    if chrom:
        impute.extend(["--chrom", chrom])
    sid = sample_id_from_filename(query)
    ids = _manifest_ids(manifest)
    if sid in ids:
        impute.extend(["--sample-id", sid])
    if tool_main(impute) != 0:
        raise SystemExit("impute failed")
    rows = _read_tsv(out / "imputed.tsv")
    report = json.loads((out / "imputed.tsv.run.json").read_text())
    write_bed(out / "imputed.bed.gz", rows, preset, report)
    stats, by_cov, by_sd = summarize(rows, report, max_cov)
    _write_tsv(out / "stats.tsv", stats)
    _write_tsv(out / "stats_by_coverage.tsv", by_cov)
    _write_tsv(out / "stats_by_sd.tsv", by_sd)
    (out / "dashboard.html").write_text(render_dashboard(stats, by_cov, by_sd, preset))


def write_bed(path: Path, rows: list[dict], preset: str, report: dict) -> None:
    kappas = []
    for chrom, info in report.get("chromosomes", {}).items():
        if "kappa" in info:
            kappas.append(f"{chrom}={float(info['kappa']):.4f}")
    header = f"# wgbs_impute preset={preset} kappa={','.join(kappas)}\n"
    lines = [header]
    for row in rows:
        pct = row.get("pct") or row.get("pct_obs") or "0"
        lines.append(
            f"{row['chrom']}\t{row['start']}\t{row['end']}\t.\t0\t.\t"
            f"{row['start']}\t{row['end']}\t0,0,0\t{row['cov']}\t{pct}\n"
        )
    with gzip.open(path, "wt") as fh:
        fh.writelines(lines)


def summarize(rows: list[dict], report: dict, max_cov: float):
    chroms = []
    for chrom in _ordered(rows):
        chroms.append(chrom)
    stats = [_chrom_stats(rows, chrom, report, max_cov) for chrom in chroms]
    stats.append(_chrom_stats(rows, "all", report, max_cov))
    by_cov = []
    by_sd = []
    for chrom in chroms + ["all"]:
        subset = rows if chrom == "all" else [r for r in rows if r["chrom"] == chrom]
        for name in COVERAGE_BINS:
            by_cov.append(_coverage_row(chrom, name, subset))
        for name in SD_BINS:
            by_sd.append(_sd_row(chrom, name, subset))
    return stats, by_cov, by_sd


def _chrom_stats(rows, chrom, report, max_cov):
    subset = rows if chrom == "all" else [r for r in rows if r["chrom"] == chrom]
    cov = np.array([float(r["cov"]) for r in subset], dtype=np.float64)
    pct = _nums(subset, "pct")
    obs = _nums(subset, "pct_obs")
    sd = _nums(subset, "panel_sd")
    sources = [r["source"] for r in subset]
    info = report.get("chromosomes", {}).get(chrom, {})
    if chrom == "all":
        infos = list(report.get("chromosomes", {}).values())
        kappa = float(np.mean([i["kappa"] for i in infos])) if infos else float("nan")
        n_solved = int(sum(i.get("n_solved", 0) for i in infos))
        n_refs = int(infos[0]["n_libraries"]) if infos else 0
    else:
        kappa = float(info.get("kappa", float("nan")))
        n_solved = int(info.get("n_solved", sources.count("blup") + sources.count("blup_fused")))
        n_refs = int(info.get("n_libraries", 0))
    positive = cov > 0
    low = (cov >= 1) & (cov < max_cov)
    return {
        "chrom": chrom,
        "n_sites": len(subset),
        "n_imputed": sources.count("blup") + sources.count("blup_fused"),
        "n_observed": sources.count("observed"),
        "n_panel_mean": sources.count("panel_mean"),
        "n_passthrough": sources.count("passthrough"),
        "n_solved": n_solved,
        "n_refs": n_refs,
        "kappa": _num(kappa),
        "query_mean_cov": _num(float(cov[positive].mean()) if positive.any() else float("nan")),
        "frac_cov0": _num(float((cov == 0).mean()) if len(cov) else float("nan")),
        "frac_cov_below_max": _num(float((cov < max_cov).mean()) if len(cov) else float("nan")),
        "mean_pct_imputed": _num(_nanmean(pct)),
        "mean_pct_observed": _num(_nanmean(obs)),
        "mean_abs_imputed_minus_observed": _num(_absdiff(pct, obs, low)),
        "median_panel_sd": _num(float(np.nanmedian(sd)) if np.isfinite(sd).any() else float("nan")),
    }


def _coverage_row(chrom, name, rows):
    cov = np.array([float(r["cov"]) for r in rows]) if rows else np.array([])
    mask = _cov_mask(cov, name)
    pct = _nums(rows, "pct")
    obs = _nums(rows, "pct_obs")
    have_obs = mask & np.isfinite(obs) & (cov >= 1)
    return {
        "chrom": chrom,
        "bin": name,
        "n_sites": int(mask.sum()),
        "mean_pct_imputed": _num(_nanmean(pct[mask]) if mask.any() else float("nan")),
        "mean_abs_imputed_minus_observed": _num(_absdiff(pct, obs, have_obs)),
    }


def _sd_row(chrom, name, rows):
    sd = _nums(rows, "panel_sd")
    mask = _sd_mask(sd, name)
    sources = np.array([r["source"] for r in rows]) if rows else np.array([])
    n = int(mask.sum())
    frac = float((sources[mask] == "panel_mean").mean()) if n else float("nan")
    return {"chrom": chrom, "bin": name, "n_sites": n, "frac_panel_mean": _num(frac)}


def _cov_mask(cov, name):
    if name == "0":
        return cov == 0
    if name == "1":
        return cov == 1
    if name == "2-4":
        return (cov >= 2) & (cov <= 4)
    if name == "5-9":
        return (cov >= 5) & (cov <= 9)
    return cov >= 10


def _sd_mask(sd, name):
    if name == "0-5":
        return np.isfinite(sd) & (sd < 5)
    if name == "5-10":
        return (sd >= 5) & (sd < 10)
    if name == "10-15":
        return (sd >= 10) & (sd < 15)
    return sd >= 15


def render_dashboard(stats, by_cov, by_sd, preset: str) -> str:
    overall = next(row for row in stats if row["chrom"] == "all")
    kappas = ", ".join(f"{row['chrom']} {row['kappa']}" for row in stats if row["chrom"] != "all")
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
<p class="note">Preset {preset}. References {overall['n_refs']}. κ {kappas}.</p>
<p>Sites written {overall['n_sites']}. Solved {overall['n_solved']}.
Panel mean {overall['n_panel_mean']}.</p>
<h2>Source</h2>
{_bars(sources)}
<h2>Imputed minus observed, by query coverage</h2>
{_bars([(r['bin'], _f(r['mean_abs_imputed_minus_observed'])) for r in cov_rows])}
<h2>Fraction left on the panel mean, by panel SD</h2>
{_bars([(r['bin'], _f(r['frac_panel_mean']) * 100) for r in sd_rows], percent=True)}
</body></html>
"""


def _bars(pairs, percent=False) -> str:
    vals = [float(v) for _, v in pairs]
    top = max(vals) if vals and max(vals) > 0 else 1.0
    lines = []
    for label, value in pairs:
        width = 100.0 * float(value) / top
        suffix = f"{float(value):.1f}%" if percent else f"{float(value):.2f}"
        lines.append(
            f'<div class="row"><span>{label}</span>'
            f'<div class="bar" style="width:{width:.1f}%"></div><span>{suffix}</span></div>'
        )
    return "\n".join(lines)


def run_demo(out: Path) -> int:
    import tempfile
    rng = np.random.default_rng(0)
    n_sites, n_libs = 400, 12
    pos = np.arange(n_sites) * 100
    z = rng.uniform(0.15, 0.85, size=n_libs)
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
        run_pipeline(refs, query, out, preset="research", max_cov=10, only_below=False,
                     chrom=None, n_refs=0, disease_role="ref", method="blup")
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
    rng_cov = np.random.default_rng(seed)
    rng_meth = np.random.default_rng(seed + 555)
    c = np.asarray(cov, dtype=np.int64)
    m = np.minimum(np.asarray(meth, dtype=np.int64), c)
    keep = min(1.0, depth / max(float(mean_full), 1e-9))
    new_c = rng_cov.binomial(c, keep)
    new_m = np.zeros_like(new_c)
    ok = new_c > 0
    new_m[ok] = rng_meth.hypergeometric(m[ok], (c - m)[ok], new_c[ok])
    return new_m, new_c


def _write_bed(path: Path, chrom, pos, meth, cov) -> None:
    lines = []
    for s, m, c in zip(pos, meth, cov):
        pct = 0.0 if c <= 0 else 100.0 * float(m) / float(c)
        start, end = int(s), int(s) + 1
        lines.append(
            f"{chrom}\t{start}\t{end}\t.\t0\t.\t{start}\t{end}\t0,0,0\t{int(c)}\t{pct:.4f}\n"
        )
    with gzip.open(path, "wt") as fh:
        fh.writelines(lines)


def _read_tsv(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:] if line]


def _write_tsv(path: Path, rows: list[dict]) -> None:
    header = list(rows[0])
    lines = ["\t".join(header)]
    lines.extend("\t".join(str(row[k]) for k in header) for row in rows)
    path.write_text("\n".join(lines) + "\n")


def _manifest_ids(path: Path) -> set[str]:
    rows = _read_tsv(path)
    return {row["sample_id"] for row in rows}


def _ordered(rows):
    seen = []
    for row in rows:
        if row["chrom"] not in seen:
            seen.append(row["chrom"])
    return seen


def _nums(rows, key):
    out = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        if row.get(key):
            out[i] = float(row[key])
    return out


def _nanmean(values):
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else float("nan")


def _absdiff(pct, obs, mask):
    sel = mask & np.isfinite(pct) & np.isfinite(obs)
    if not sel.any():
        return float("nan")
    return float(np.mean(np.abs(pct[sel] - obs[sel])))


def _num(value) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.6g}"


def _f(text) -> float:
    return float(text) if text else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
