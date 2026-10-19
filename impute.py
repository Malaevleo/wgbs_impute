"""Impute one query from a prepared panel cache."""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from wgbs_impute import __version__
from wgbs_impute.blup import blup_gates, context_one, fuse
from wgbs_impute.io import in_regions, mean_positive_cov, read_regions, read_sites
from wgbs_impute.panel import _fit_chrom, choose_refs, load_chrom, load_root, prepare_panel

CORE = ("blup", "panel_mean", "neighbor_mean")
OPTIONAL = ("local", "global", "comet", "hmm", "kalman")
HEADER = [
    "chrom", "start", "end", "cov", "meth", "pct_obs", "pct",
    "pct_blup", "pct_fused", "pct_panel", "n_neighbors", "panel_sd", "source",
]


def impute_query(
    query: Path,
    output: Path,
    *,
    panel_cache: Path | None = None,
    panel_manifest: Path | None = None,
    fmt: str | None = None,
    chrom: str | None = None,
    regions: Path | None = None,
    max_cov: float = 10,
    only_below: bool = False,
    score_all: bool = False,
    sample_id: str | None = None,
    min_ref_mean_cov: float = 10,
    min_ref_site_cov: int = 10,
    n_refs: int = 0,
    window: int = 25000,
    neighbors: int = 30,
    shrink: float = 0.25,
    noise_floor: float = 0.05,
    neighbor_min_cov: int = 1,
    sd_passthrough: float = 5,
    fuse_on: bool = True,
    kappa=None,
    threads: int = 1,
    seed: int = 42,
    preset: str = "pairwise",
    methods: list[str] | None = None,
    local_iterations: int = 120,
    local_depth: int = 4,
    max_train: int = 20000,
    refnn_k: int = 5,
) -> dict:
    t0 = time.time()
    methods = _methods(methods)
    if any(m in OPTIONAL for m in methods):
        print("CatBoost, Comet, HMM, and Kalman run per site and will dominate runtime.", flush=True)
    cache = Path(panel_cache) if panel_cache else Path(panel_manifest).with_name("panel_cache")
    if not (cache / "meta.json").exists():
        if panel_manifest is None:
            raise SystemExit(f"no panel cache at {cache}")
        prepare_panel(
            panel_manifest, cache, min_ref_mean_cov=min_ref_mean_cov,
            min_ref_site_cov=min_ref_site_cov, window=window, neighbors=neighbors,
            shrink=shrink,             sd_passthrough=sd_passthrough, seed=seed, preset=preset,
        )
    root = load_root(cache)
    region_map = read_regions(regions) if regions else None
    query_sites = read_sites(Path(query), fmt, chrom)
    if chrom:
        query_sites = {c: v for c, v in query_sites.items() if c == chrom}
    rows: list[list[str]] = []
    chrom_stats = {}
    used_ids: list[str] = []
    for ch, tab in query_sites.items():
        if ch not in root["chromosomes"] or not (cache / ch / "meta.json").exists():
            print(f"skip {ch}: not in the panel cache", flush=True)
            continue
        packed, stat, ids = _impute_chrom(
            cache, ch, tab, root, methods,
            sample_id=sample_id, n_refs=n_refs, window=window, neighbors=neighbors,
            shrink=shrink, noise_floor=noise_floor, neighbor_min_cov=neighbor_min_cov,
            sd_passthrough=sd_passthrough, max_cov=max_cov, only_below=only_below,
            score_all=score_all, fuse_on=fuse_on, kappa=kappa, regions=region_map,
            seed=seed, threads=threads, preset=preset, local_iterations=local_iterations,
            local_depth=local_depth, max_train=max_train, refnn_k=refnn_k,
        )
        rows.extend(packed)
        chrom_stats[ch] = stat
        used_ids = ids
        print(
            f"{ch}: libraries {stat['n_libraries']}  written {stat['n_written']}  "
            f"solved {stat['n_solved']}  panel_mean {stat['n_panel_mean']}  "
            f"kappa {stat['kappa']:.3f}  {stat['seconds']:.1f}s",
            flush=True,
        )
    if not rows:
        raise SystemExit("no query sites overlapped the panel")
    header = list(HEADER)
    for name in methods:
        if name != "blup":
            header.append(f"pct_{name}")
        if fuse_on and name not in ("panel_mean",):
            col = f"pct_{name}_fused" if name != "blup" else None
            if col and col not in header:
                header.append(col)
    _write_table(Path(output), header, rows)
    report = {
        "version": __version__,
        "args": _jsonable(dict(
            query=str(query), output=str(output), panel_cache=str(cache),
            methods=methods, max_cov=max_cov, fuse=fuse_on, window=window,
            neighbors=neighbors, shrink=shrink, noise_floor=noise_floor,
            sd_passthrough=sd_passthrough, n_refs=n_refs, sample_id=sample_id,
            seed=seed, threads=threads, preset=preset,
        )),
        "panel_sample_ids": used_ids,
        "chromosomes": chrom_stats,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    Path(str(output) + ".run.json").write_text(json.dumps(report, indent=2))
    return report


def _methods(names: list[str] | None) -> list[str]:
    if not names:
        return ["blup"]
    out = []
    for name in names:
        if name not in CORE + OPTIONAL:
            raise SystemExit(f"unknown method {name}")
        if name not in out:
            out.append(name)
    if "blup" not in out:
        out.insert(0, "blup")
    return out


def _impute_chrom(cache, chrom, tab, root, methods, **opt):
    t0 = time.time()
    data = load_chrom(cache, chrom)
    meta = data["meta"]
    ids = choose_refs(meta["sample_ids"], meta["groups"], meta["mean_cov"], opt["n_refs"], opt["sample_id"])
    same = (
        ids == list(range(len(meta["sample_ids"])))
        and opt["window"] == root["window"]
        and opt["neighbors"] == root["neighbors"]
        and opt["shrink"] == root["shrink"]
        and opt.get("preset", "pairwise") == root.get("preset", meta.get("preset", "pairwise"))
    )
    pos = np.asarray(data["pos"])
    if same:
        neigh = np.asarray(data["neigh"])
        factors = {k: np.asarray(data[k]) for k in
                   ("panel_mean", "panel_sd", "n_label", "mu", "mu_n", "sin", "snn", "solvable")}
        beta = np.asarray(data["beta10"])
        meth_p = np.asarray(data["meth"])
        cov_p = np.asarray(data["cov"])
    else:
        beta = np.asarray(data["beta10"][:, ids])
        beta1 = np.asarray(data["beta1"][:, ids])
        meth_p = np.asarray(data["meth"][:, ids])
        cov_p = np.asarray(data["cov"][:, ids])
        min_ok, min_ov = blup_gates(len(ids))
        neigh, factors = _fit_chrom(
            pos, beta, beta1, opt["window"], opt["neighbors"], opt["shrink"],
            min_ok, min_ov, opt.get("preset", "pairwise"),
        )
    min_ok, _ = blup_gates(beta.shape[1] if not same else len(meta["sample_ids"]))
    q_pos = tab["start"]
    loc = np.searchsorted(pos, q_pos)
    matched = (loc < len(pos)) & (pos[np.clip(loc, 0, len(pos) - 1)] == q_pos)
    q_beta = np.full(len(pos), np.nan)
    q_meth = np.zeros(len(pos))
    q_cov = np.zeros(len(pos))
    hit = loc[matched]
    q_meth[hit] = tab["meth"][matched]
    q_cov[hit] = tab["cov"][matched]
    seen = q_cov >= opt["neighbor_min_cov"]
    q_beta[seen] = q_meth[seen] / np.maximum(q_cov[seen], 1.0)
    kappa = _kappa_for(meta, mean_positive_cov(tab["cov"]), opt["kappa"])
    need = []
    for i, start in enumerate(q_pos):
        if opt["regions"] is not None and not in_regions(chrom, int(start), opt["regions"]):
            continue
        if not matched[i]:
            continue
        if float(tab["cov"][i]) >= opt["max_cov"] and not opt["score_all"]:
            continue
        need.append(int(loc[i]))
    extra = _optional(
        methods, pos, beta, meth_p, cov_p, neigh, q_beta, q_meth, q_cov, factors,
        opt, min_ok, np.asarray(need, dtype=np.int64),
    )
    rows = []
    n_solved = n_panel = n_written = 0
    sd_cut = opt["sd_passthrough"]
    for i, start in enumerate(q_pos):
        if opt["regions"] is not None and not in_regions(chrom, int(start), opt["regions"]):
            continue
        cov = float(tab["cov"][i])
        meth = float(tab["meth"][i])
        end = int(tab["end"][i])
        if not matched[i] or not np.isfinite(factors["panel_mean"][loc[i]]):
            if opt["only_below"] and cov >= opt["max_cov"]:
                continue
            rows.append(_row(chrom, start, end, cov, meth, {}, "passthrough", methods, opt["fuse_on"]))
            n_written += 1
            continue
        j = int(loc[i])
        low = cov < opt["max_cov"]
        if not low and not opt["score_all"]:
            if opt["only_below"]:
                continue
            rows.append(_row(chrom, start, end, cov, meth, _blank_pred(factors, j), "observed", methods, opt["fuse_on"], observed=True))
            n_written += 1
            continue
        pred, n_nb, solved = _context_at(
            j, neigh, factors, q_beta, q_cov, min_ok, opt["noise_floor"], sd_cut,
            clip_query=opt.get("preset") == "research",
        )
        stable = sd_cut > 0 and float(factors["panel_sd"][j]) < sd_cut
        if solved:
            n_solved += 1
        else:
            n_panel += 1
        fused = float(fuse(np.array([pred]), np.array([meth]), np.array([cov]), kappa)[0])
        if not low:
            source = "observed"
        elif stable:
            source = "panel_mean"
        elif solved:
            source = "blup_fused" if opt["fuse_on"] else "blup"
        else:
            source = "panel_mean"
        values = {
            "pct_blup": pred, "pct_fused": fused, "pct_panel": float(factors["panel_mean"][j]),
            "n_neighbors": n_nb, "panel_sd": float(factors["panel_sd"][j]), "stable": stable,
        }
        for name, vec in extra.items():
            values[f"pct_{name}"] = vec[j] if np.isfinite(vec[j]) else ""
            if opt["fuse_on"] and name != "panel_mean":
                fv = fuse(np.array([vec[j]]), np.array([meth]), np.array([cov]), kappa)[0]
                values[f"pct_{name}_fused"] = float(fv) if np.isfinite(vec[j]) else ""
        if "panel_mean" in methods:
            values["pct_panel_mean"] = float(factors["panel_mean"][j])
        if "neighbor_mean" in methods:
            from wgbs_impute.optional import neighbor_mean
            nm, _ = neighbor_mean(q_beta, neigh[j], float(factors["panel_mean"][j]))
            values["pct_neighbor_mean"] = nm
            if opt["fuse_on"]:
                values["pct_neighbor_mean_fused"] = float(
                    fuse(np.array([nm]), np.array([meth]), np.array([cov]), kappa)[0]
                )
        rows.append(_row(chrom, start, end, cov, meth, values, source, methods, opt["fuse_on"], observed=not low))
        n_written += 1
    stat = {
        "kappa": kappa, "n_written": n_written, "n_solved": n_solved,
        "n_panel_mean": n_panel, "n_libraries": len(ids), "seconds": time.time() - t0,
    }
    return rows, stat, [meta["sample_ids"][i] for i in ids]


def _context_at(j, neigh, factors, q_beta, q_cov, min_ok, floor, sd_cut, clip_query=False):
    nb = neigh[j]
    nb = nb[nb >= 0]
    sd = float(factors["panel_sd"][j])
    mean = float(factors["panel_mean"][j])
    if sd_cut > 0 and sd < sd_cut:
        return mean, 0, False
    return context_one(
        float(factors["mu"][j]) if np.isfinite(factors["mu"][j]) else np.nan,
        np.asarray(factors["mu_n"][j, : len(nb)], dtype=np.float64),
        np.asarray(factors["sin"][j, : len(nb)], dtype=np.float64),
        np.asarray(factors["snn"][j, : len(nb), : len(nb)], dtype=np.float64),
        q_beta[nb], q_cov[nb], int(factors["n_label"][j]), min_ok, floor, mean,
        bool(np.isfinite(factors["mu"][j]) and len(nb) >= 2 and int(factors["n_label"][j]) >= min_ok),
        clip_query=clip_query,
    )


def _optional(methods, pos, beta, meth_p, cov_p, neigh, q_beta, q_meth, q_cov, factors, opt, min_ok, need):
    from wgbs_impute import optional
    hide = q_cov < opt["max_cov"]
    out = {}
    if "local" in methods and len(need):
        pred = np.full(len(pos), np.nan)
        pred[need] = optional.predict_local(
            beta, neigh, q_beta, need, iterations=opt["local_iterations"],
            depth=opt["local_depth"], seed=opt["seed"], min_ok=min_ok,
        )
        miss = ~np.isfinite(pred)
        pred[miss] = factors["panel_mean"][miss]
        out["local"] = pred
    if "global" in methods and len(need):
        pred = np.full(len(pos), np.nan)
        pred[need] = optional.predict_global(
            pos, meth_p.astype(np.float64), cov_p.astype(np.float64), beta,
            q_meth, q_cov, need, max_train=opt["max_train"],
            refnn_k=opt["refnn_k"], seed=opt["seed"], threads=max(opt["threads"], 1),
        )
        miss = ~np.isfinite(pred)
        pred = np.where(miss, factors["panel_mean"], pred)
        out["global"] = pred
    if "comet" in methods:
        out["comet"] = optional.predict_comet(
            pos, beta, q_meth, q_cov, factors["panel_mean"], hide, opt["seed"],
        )
    if "hmm" in methods:
        out["hmm"] = optional.predict_hmm(pos, beta, q_meth, q_cov, hide)
    if "kalman" in methods:
        out["kalman"] = optional.predict_kalman(
            pos, factors["panel_mean"], factors["panel_sd"], q_meth, q_cov, hide,
        )
    return out


def _kappa_for(meta, query_mean, override):
    if override is not None:
        if isinstance(override, (int, float)):
            return float(override)
        depths, values = override
        return float(np.interp(query_mean, depths, values))
    depths = np.asarray(meta["kappa_depths"], dtype=np.float64)
    values = np.asarray(meta["kappa"], dtype=np.float64)
    if query_mean <= 0:
        return float(values[0])
    return float(np.interp(query_mean, depths, values))


def _blank_pred(factors, j):
    return {
        "pct_panel": float(factors["panel_mean"][j]),
        "panel_sd": float(factors["panel_sd"][j]),
        "n_neighbors": 0,
    }


def _row(chrom, start, end, cov, meth, values, source, methods, fuse_on, observed=False):
    pct_obs = "" if cov <= 0 else f"{100.0 * meth / cov:.4f}"
    decision = ""
    if source == "observed":
        decision = pct_obs
    elif source == "passthrough":
        decision = pct_obs
    elif source == "panel_mean" and values.get("stable"):
        decision = _fmt(values.get("pct_panel"))
    elif source == "panel_mean":
        decision = _fmt(values.get("pct_fused") if fuse_on else values.get("pct_panel"))
    elif source == "blup":
        decision = _fmt(values.get("pct_blup"))
    else:
        decision = _fmt(values.get("pct_fused"))
    base = [
        chrom, str(int(start)), str(int(end)), str(int(round(cov))), str(int(round(meth))),
        pct_obs, decision, _fmt(values.get("pct_blup")), _fmt(values.get("pct_fused")),
        _fmt(values.get("pct_panel")), str(int(values.get("n_neighbors", 0) or 0)),
        _fmt(values.get("panel_sd")), source,
    ]
    for name in methods:
        if name == "blup":
            continue
        base.append(_fmt(values.get(f"pct_{name}")))
        if fuse_on and name != "panel_mean":
            base.append(_fmt(values.get(f"pct_{name}_fused")))
    return base


def _fmt(value):
    if value is None or value == "" or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    return f"{float(value):.4f}"


def _write_table(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\t".join(header) + "\n" + "\n".join("\t".join(r) for r in rows) + "\n"
    if path.suffix != ".gz":
        path.write_text(text)
        return
    raw = path.with_suffix("")
    raw.write_text(text)
    bgzip = shutil.which("bgzip")
    if bgzip:
        subprocess.run([bgzip, "-f", str(raw)], check=True)
        produced = Path(str(raw) + ".gz")
        if produced != path:
            produced.replace(path)
        return
    with gzip.open(path, "wt") as fh:
        fh.write(text)
    raw.unlink(missing_ok=True)


def _jsonable(obj):
    out = {}
    for k, v in obj.items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def parse_kappa_arg(text: str | None):
    if text is None:
        return None
    path = Path(text)
    if path.exists():
        depths, values = [], []
        for line in path.read_text().splitlines():
            if not line.strip() or line.startswith("depth"):
                continue
            a, b = line.split()
            depths.append(float(a))
            values.append(float(b))
        order = np.argsort(depths)
        return (np.asarray(depths)[order], np.asarray(values)[order])
    return float(text)
