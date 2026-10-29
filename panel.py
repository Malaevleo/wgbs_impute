"""Prepare versioned, portable chromosome caches and calibrate fusion weights."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from wgbs_impute import __version__
from wgbs_impute.blup import (
    blup_gates,
    calibrate,
    context_one,
    factor_row,
    panel_summary,
    research_neighbors,
    select_neighbors,
)
from wgbs_impute.io import read_manifest, read_sites
from wgbs_impute.thinning import subsample_depths
from wgbs_impute.validation import validate_chrom, validate_options

CACHE_SCHEMA = 3
KAPPA_DEPTHS = (2.0, 5.0, 10.0, 20.0)


def prepare_panel(
    manifest: Path,
    cache: Path,
    *,
    min_ref_mean_cov: float = 10.0,
    min_ref_site_cov: int = 10,
    window: int = 25000,
    neighbors: int = 30,
    shrink: float = 0.25,
    sd_passthrough: float | None = None,
    seed: int = 42,
    kappa_sites: int = 2000,
    preset: str = "pairwise",
) -> dict:
    if sd_passthrough is None:
        sd_passthrough = 0.0 if preset == "research" else 5.0
    validate_options(**locals())
    rows = read_manifest(Path(manifest))
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        raise ValueError(f"cache already exists: {cache}; choose a new directory")
    lock = cache.with_name(cache.name + ".lock")
    try:
        lock_file = lock.open("x")
    except FileExistsError as exc:
        raise ValueError(f"cache preparation lock exists: {lock}") from exc
    try:
        with lock_file, tempfile.TemporaryDirectory(prefix=f".{cache.name}-", dir=cache.parent) as tmp:
            stage = Path(tmp) / "cache"
            stage.mkdir()
            libraries = {}
            # Parse each input once, spool chromosome arrays, release its row dictionary.
            for number, row in enumerate(rows):
                sites = read_sites(row["path"], chrom=row["chrom"])
                if row["chrom"]:
                    sites = {ch: tab for ch, tab in sites.items() if ch == row["chrom"]}
                    if not sites:
                        raise ValueError(f"{row['path']}: no rows for {row['chrom']}")
                sid = row["sample_id"]
                lib = libraries.setdefault(
                    sid, {**row, "sites": {}, "cov_sum": 0.0, "cov_count": 0, "sources": []}
                )
                if lib["group"] != row["group"] or lib["mean_cov"] != row["mean_cov"]:
                    raise ValueError(f"inconsistent metadata across files for {sid}")
                for ch, tab in sites.items():
                    if ch in lib["sites"]:
                        raise ValueError(f"overlapping files for sample {sid}, chromosome {ch}")
                    dest = Path(tmp) / f"input-{number}-{len(lib['sites'])}.npz"
                    np.savez(dest, **tab)
                    lib["sites"][ch] = dest
                    positive = tab["cov"][tab["cov"] > 0]
                    lib["cov_sum"] += float(positive.sum())
                    lib["cov_count"] += len(positive)
                stat = row["path"].stat()
                lib["sources"].append(
                    {"path": str(row["path"].resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                )
                del sites
            kept = []
            for lib in libraries.values():
                computed = lib["cov_sum"] / max(lib["cov_count"], 1)
                lib["mean_cov"] = computed if lib["mean_cov"] is None else lib["mean_cov"]
                if not np.isfinite(lib["mean_cov"]) or lib["mean_cov"] < 0:
                    raise ValueError("reference mean coverage must be finite and nonnegative")
                if lib["mean_cov"] <= min_ref_mean_cov:
                    print(f"drop {lib['sample_id']}: mean coverage {lib['mean_cov']:.2f}x", flush=True)
                else:
                    kept.append(lib)
            if len(kept) < 2:
                raise ValueError("need at least two libraries above the coverage cutoff")
            chroms = sorted({ch for lib in kept for ch in lib["sites"]})
            chroms = [ch for ch in chroms if sum(ch in lib["sites"] for lib in kept) >= 2]
            if not chroms:
                raise ValueError("no chromosome has at least two reference libraries")
            meta = {
                "schema": CACHE_SCHEMA,
                "version": __version__,
                "sample_ids": [lib["sample_id"] for lib in kept],
                "groups": [lib["group"] for lib in kept],
                "mean_cov": [lib["mean_cov"] for lib in kept],
                "sources": [lib["sources"] for lib in kept],
                "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                "window": window,
                "neighbors": neighbors,
                "shrink": shrink,
                "min_ref_site_cov": min_ref_site_cov,
                "sd_passthrough": sd_passthrough,
                "seed": seed,
                "kappa_sites": kappa_sites,
                "preset": preset,
                "chromosomes": chroms,
            }
            for ch in chroms:
                _prepare_chrom(
                    stage,
                    ch,
                    kept,
                    min_ref_site_cov,
                    window,
                    neighbors,
                    shrink,
                    sd_passthrough,
                    seed,
                    kappa_sites,
                    preset,
                )
                print(f"prepared {ch}", flush=True)
            # Publish only a complete cache. A failed preparation never looks valid.
            (stage / "meta.json").write_text(json.dumps(meta, indent=2, allow_nan=False))
            stage.rename(cache)
            return meta
    finally:
        lock.unlink(missing_ok=True)


def _prepare_chrom(
    cache,
    chrom,
    libraries,
    min_site_cov,
    window,
    neighbors,
    shrink,
    sd_passthrough,
    seed,
    kappa_sites,
    preset,
):
    present = []
    for lib in libraries:
        if chrom in lib["sites"]:
            with np.load(lib["sites"][chrom], allow_pickle=False) as archive:
                present.append({**lib, "table": {k: archive[k] for k in archive.files}})
    pos = np.unique(np.concatenate([lib["table"]["start"] for lib in present]))
    n, r = len(pos), len(present)
    beta10 = np.full((n, r), np.nan, dtype=np.float32)
    beta1 = np.full((n, r), np.nan, dtype=np.float32)
    meth = np.zeros((n, r), dtype=np.uint32)
    cov = np.zeros((n, r), dtype=np.uint32)
    for j, lib in enumerate(present):
        tab = lib.pop("table")
        idx = np.searchsorted(pos, tab["start"])
        cov[idx, j] = tab["cov"].astype(np.uint32)
        meth[idx, j] = tab["meth"].astype(np.uint32)
        np.divide(meth[:, j], cov[:, j], out=beta1[:, j], where=cov[:, j] >= 1)
        np.divide(meth[:, j], cov[:, j], out=beta10[:, j], where=cov[:, j] >= min_site_cov)
    min_ok, min_ov = blup_gates(r)
    neigh, factors = _fit_chrom(pos, beta10, beta1, window, neighbors, shrink, min_ok, min_ov, preset)
    dest = cache / chrom
    dest.mkdir()
    arrays = {
        "pos": pos,
        "beta10": beta10,
        "beta1": beta1,
        "meth": meth,
        "cov": cov,
        "neigh": neigh,
        **{k: factors[k] for k in ("panel_mean", "panel_sd", "n_label")},
    }
    for name, array in arrays.items():
        np.save(dest / f"{name}.npy", array, allow_pickle=False)
    kappa = _kappa_grid(
        pos,
        beta10,
        beta1,
        meth,
        cov,
        present,
        window,
        neighbors,
        shrink,
        sd_passthrough,
        seed,
        kappa_sites,
        preset,
    )
    (dest / "meta.json").write_text(
        json.dumps(
            {
                "sample_ids": [lib["sample_id"] for lib in present],
                "groups": [lib["group"] for lib in present],
                "mean_cov": [lib["mean_cov"] for lib in present],
                "kappa_depths": list(KAPPA_DEPTHS),
                "kappa": kappa,
                "preset": preset,
            },
            allow_nan=False,
        )
    )


def _fit_chrom(pos, beta10, beta1, window, neighbors, shrink, min_ok, min_ov, preset, sites=None):
    selector = research_neighbors if preset == "research" else select_neighbors
    neigh = selector(pos, beta10, window, neighbors, min_ov, sites=sites)
    factors = panel_summary(beta10)
    factors["solvable"] = (factors["n_label"] >= min_ok) & ((neigh >= 0).sum(axis=1) >= 2)
    factors["model"] = (
        beta10,
        beta1 if preset == "research" else None,
        neigh,
        shrink,
        min_ok,
        preset == "research",
    )
    return neigh, factors


def _kappa_grid(
    pos,
    beta10,
    beta1,
    meth,
    cov,
    present,
    window,
    neighbors,
    shrink,
    sd_passthrough,
    seed,
    kappa_sites,
    preset,
):
    hold = int(np.argmax([lib["mean_cov"] for lib in present]))
    keep = [j for j in range(beta10.shape[1]) if j != hold]
    if len(keep) < 2:
        return [8.0] * len(KAPPA_DEPTHS)
    beta, beta_x = beta10[:, keep], beta1[:, keep]
    min_ok, min_ov = blup_gates(len(keep))
    summary = panel_summary(beta)
    eligible = np.flatnonzero(
        (summary["panel_sd"] >= sd_passthrough) & (summary["n_label"] >= min_ok) & (cov[:, hold] >= 10)
    )
    if len(eligible) > kappa_sites:
        order = np.argsort(summary["panel_sd"][eligible], kind="stable")
        eligible = eligible[order[np.linspace(0, len(order) - 1, kappa_sites).astype(int)]]
    if len(eligible) == 0:
        return [8.0] * len(KAPPA_DEPTHS)
    neigh, factors = _fit_chrom(
        pos, beta, beta_x, window, neighbors, shrink, min_ok, min_ov, preset, sites=eligible
    )
    eligible = eligible[factors["solvable"][eligible]]
    if len(eligible) == 0:
        return [8.0] * len(KAPPA_DEPTHS)
    full_c, full_m = cov[:, hold].astype(float), meth[:, hold].astype(float)
    truth = np.divide(100 * full_m, full_c, out=np.full(len(pos), np.nan), where=full_c > 0)
    mean_full = float(full_c[full_c > 0].mean()) if np.any(full_c > 0) else 0.0
    drawn = subsample_depths(full_m, full_c, KAPPA_DEPTHS, mean_full, seed)
    pred = np.full((len(KAPPA_DEPTHS), len(eligible)), np.nan)
    for t, i in enumerate(eligible):
        nb = neigh[i][neigh[i] >= 0]
        row = factor_row(factors, i)
        for d, depth in enumerate(KAPPA_DEPTHS):
            m, c = drawn[depth]
            q_beta = np.divide(m[nb], c[nb], out=np.full(len(nb), np.nan), where=c[nb] >= 1)
            pred[d, t], _, _ = context_one(
                float(row["mu"][0]),
                row["mu_n"][0, : len(nb)].astype(float),
                row["sin"][0, : len(nb)].astype(float),
                row["snn"][0, : len(nb), : len(nb)].astype(float),
                q_beta,
                c[nb],
                int(row["n_label"][0]),
                min_ok,
                0.05,
                float(row["panel_mean"][0]),
                bool(row["solvable"][0]),
                clip_query=preset == "research",
            )
    return [calibrate(p, truth[eligible]) for p in pred]


def load_root(cache: Path) -> dict:
    meta = json.loads((Path(cache) / "meta.json").read_text())
    if meta.get("schema") != CACHE_SCHEMA:
        raise ValueError("incompatible panel cache; rebuild it with prepare-panel")
    for chrom in meta["chromosomes"]:
        validate_chrom(chrom)
    return meta


def load_chrom(cache: Path, chrom: str) -> dict:
    validate_chrom(chrom)
    dest = Path(cache) / chrom
    meta = json.loads((dest / "meta.json").read_text())
    names = ("pos", "beta10", "beta1", "meth", "cov", "panel_mean", "panel_sd", "n_label", "neigh")
    data = {
        "meta": meta,
        **{name: np.load(dest / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in names},
    }
    n, r = len(data["pos"]), len(meta["sample_ids"])
    if not n or r < 2 or len(set(meta["sample_ids"])) != r:
        raise ValueError(f"{chrom}: invalid reference identities or empty cache; rebuild the panel")
    expected = {name: (n, r) for name in ("beta10", "beta1", "meth", "cov")}
    expected.update({name: (n,) for name in ("pos", "panel_mean", "panel_sd", "n_label")})
    if any(data[name].shape != shape for name, shape in expected.items()):
        raise ValueError(f"{chrom}: inconsistent cache array shapes; rebuild the panel")
    if data["neigh"].ndim != 2 or data["neigh"].shape[0] != n or np.any(np.diff(data["pos"]) <= 0):
        raise ValueError(f"{chrom}: invalid positions or neighbor array; rebuild the panel")
    return data


def choose_refs(sample_ids, groups, mean_cov, n_refs: int, exclude: str | None) -> list[int]:
    idx = [i for i, sid in enumerate(sample_ids) if sid != exclude]
    if n_refs and n_refs < len(idx):
        order = sorted(idx, key=lambda i: -mean_cov[i])
        by = {}
        for i in order:
            by.setdefault(groups[i] or "ungrouped", []).append(i)
        chosen = []
        while len(chosen) < n_refs and any(by.values()):
            for group in sorted(by):
                if by[group] and len(chosen) < n_refs:
                    chosen.append(by[group].pop(0))
        return chosen
    return idx
