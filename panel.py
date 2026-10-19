"""Build a per-chromosome panel cache. One chromosome is never read to prepare another."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from wgbs_impute import __version__
from wgbs_impute.blup import (
    blup_gates,
    calibrate,
    context_one,
    select_neighbors,
    site_factors,
)
from wgbs_impute.io import mean_positive_cov, read_manifest, read_sites

KAPPA_DEPTHS = (2.0, 5.0, 10.0, 20.0)


def _thin():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from audit_fix.thinning import subsample_depths
    return subsample_depths


def prepare_panel(
    manifest: Path,
    cache: Path,
    *,
    min_ref_mean_cov: float = 10.0,
    min_ref_site_cov: int = 10,
    window: int = 25000,
    neighbors: int = 30,
    shrink: float = 0.25,
    sd_passthrough: float = 5.0,
    seed: int = 42,
    kappa_sites: int = 2000,
) -> dict:
    rows = read_manifest(Path(manifest))
    libraries = []
    for row in rows:
        sites = read_sites(row["path"], chrom=row["chrom"])
        covs = [mean_positive_cov(ch["cov"]) for ch in sites.values()]
        computed = float(np.average(covs)) if covs else 0.0
        mean_cov = row["mean_cov"] if row["mean_cov"] is not None else computed
        if mean_cov <= min_ref_mean_cov:
            print(f"drop {row['sample_id']}: mean coverage {mean_cov:.2f}x", flush=True)
            continue
        libraries.append({**row, "sites": sites, "mean_cov": mean_cov, "computed_mean_cov": computed})
    if len(libraries) < 2:
        raise SystemExit("need at least two libraries above the coverage cutoff")
    chroms = sorted({ch for lib in libraries for ch in lib["sites"]})
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": __version__,
        "sample_ids": [lib["sample_id"] for lib in libraries],
        "groups": [lib["group"] for lib in libraries],
        "mean_cov": [lib["mean_cov"] for lib in libraries],
        "window": window,
        "neighbors": neighbors,
        "shrink": shrink,
        "min_ref_site_cov": min_ref_site_cov,
        "sd_passthrough": sd_passthrough,
        "chromosomes": chroms,
    }
    (cache / "meta.json").write_text(json.dumps(meta, indent=2))
    for chrom in chroms:
        _prepare_chrom(
            cache, chrom, libraries, min_ref_site_cov, window, neighbors, shrink,
            sd_passthrough, seed, kappa_sites,
        )
        print(f"prepared {chrom}: {len(libraries)} libraries", flush=True)
    return meta


def _prepare_chrom(
    cache, chrom, libraries, min_site_cov, window, neighbors, shrink,
    sd_passthrough, seed, kappa_sites,
):
    present = [lib for lib in libraries if chrom in lib["sites"]]
    if len(present) < 2:
        print(f"skip {chrom}: fewer than two libraries", flush=True)
        return
    starts = sorted({int(s) for lib in present for s in lib["sites"][chrom]["start"]})
    pos = np.asarray(starts, dtype=np.int64)
    index = {int(s): i for i, s in enumerate(pos)}
    n, r = len(pos), len(present)
    beta10 = np.full((n, r), np.nan, dtype=np.float32)
    beta1 = np.full((n, r), np.nan, dtype=np.float32)
    meth = np.zeros((n, r), dtype=np.uint32)
    cov = np.zeros((n, r), dtype=np.uint32)
    for j, lib in enumerate(present):
        tab = lib["sites"][chrom]
        for s, m, c in zip(tab["start"], tab["meth"], tab["cov"]):
            i = index[int(s)]
            ci = int(min(c, np.iinfo(np.uint32).max))
            mi = int(min(max(m, 0.0), ci))
            meth[i, j] = mi
            cov[i, j] = ci
            if ci >= 1:
                beta1[i, j] = mi / ci
            if ci >= min_site_cov:
                beta10[i, j] = mi / ci
    min_ok, min_ov = blup_gates(r)
    neigh = select_neighbors(pos, beta10, window, neighbors, min_ov)
    factors = site_factors(beta10, neigh, shrink, min_ok)
    # Stable sites do not need a stored factorization. Keep the panel mean.
    low = factors["panel_sd"] < sd_passthrough
    factors["solvable"][low] = False
    dest = cache / chrom
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "pos.npy", pos.astype(np.int32))
    np.save(dest / "beta10.npy", beta10)
    np.save(dest / "beta1.npy", beta1)
    np.save(dest / "meth.npy", meth)
    np.save(dest / "cov.npy", cov)
    np.save(dest / "panel_mean.npy", factors["panel_mean"])
    np.save(dest / "panel_sd.npy", factors["panel_sd"])
    np.save(dest / "n_label.npy", factors["n_label"])
    np.save(dest / "neigh.npy", neigh)
    np.save(dest / "mu.npy", factors["mu"])
    np.save(dest / "mu_n.npy", factors["mu_n"])
    np.save(dest / "sin.npy", factors["sin"])
    np.save(dest / "snn.npy", factors["snn"])
    np.save(dest / "solvable.npy", factors["solvable"])
    kappa = _kappa_grid(
        pos, beta10, meth, cov, present, window, neighbors, shrink,
        sd_passthrough, seed, kappa_sites,
    )
    (dest / "meta.json").write_text(json.dumps({
        "sample_ids": [lib["sample_id"] for lib in present],
        "groups": [lib["group"] for lib in present],
        "mean_cov": [lib["mean_cov"] for lib in present],
        "kappa_depths": list(KAPPA_DEPTHS),
        "kappa": kappa,
    }))


def _kappa_grid(
    pos, beta10, meth, cov, present, window, neighbors, shrink,
    sd_passthrough, seed, kappa_sites,
) -> list[float]:
    means = np.array([lib["mean_cov"] for lib in present])
    hold = int(np.argmax(means))
    keep = [j for j in range(beta10.shape[1]) if j != hold]
    if len(keep) < 2:
        return [8.0 for _ in KAPPA_DEPTHS]
    sub = beta10[:, keep]
    min_ok, min_ov = blup_gates(len(keep))
    neigh = select_neighbors(pos, sub, window, neighbors, min_ov)
    factors = site_factors(sub, neigh, shrink, min_ok)
    eligible = np.flatnonzero(
        (factors["panel_sd"] >= sd_passthrough) & factors["solvable"] & np.isfinite(factors["panel_mean"])
    )
    if len(eligible) == 0:
        return [8.0 for _ in KAPPA_DEPTHS]
    if len(eligible) > kappa_sites:
        order = np.argsort(factors["panel_sd"][eligible])
        pick = order[np.linspace(0, len(order) - 1, kappa_sites).astype(int)]
        eligible = eligible[pick]
    truth = np.full(len(pos), np.nan)
    full_c = cov[:, hold].astype(np.float64)
    full_m = meth[:, hold].astype(np.float64)
    ok = full_c >= 10
    truth[ok] = 100.0 * full_m[ok] / full_c[ok]
    mean_full = float(full_c[full_c > 0].mean()) if np.any(full_c > 0) else 0.0
    drawn = _thin()(full_m, full_c, list(KAPPA_DEPTHS), mean_full, seed)
    out = []
    for depth in KAPPA_DEPTHS:
        m, c = drawn[depth]
        pred = np.full(len(eligible), np.nan)
        for t, i in enumerate(eligible):
            nb = neigh[i]
            nb = nb[nb >= 0]
            q_beta = np.divide(m[nb], c[nb], out=np.full(len(nb), np.nan), where=c[nb] >= 1)
            pred[t], _, _ = context_one(
                float(factors["mu"][i]) if np.isfinite(factors["mu"][i]) else np.nan,
                factors["mu_n"][i, : len(nb)].astype(np.float64),
                factors["sin"][i, : len(nb)].astype(np.float64),
                factors["snn"][i, : len(nb), : len(nb)].astype(np.float64),
                q_beta, c[nb].astype(np.float64), int(factors["n_label"][i]), min_ok, 0.05,
                float(factors["panel_mean"][i]), bool(factors["solvable"][i]),
            )
        out.append(calibrate(pred, truth[eligible]))
    return out


def load_root(cache: Path) -> dict:
    return json.loads((Path(cache) / "meta.json").read_text())


def load_chrom(cache: Path, chrom: str) -> dict:
    dest = Path(cache) / chrom
    meta = json.loads((dest / "meta.json").read_text())
    def arr(name):
        return np.load(dest / name, mmap_mode="r")
    return {
        "meta": meta,
        "pos": arr("pos.npy"),
        "beta10": arr("beta10.npy"),
        "beta1": arr("beta1.npy"),
        "meth": arr("meth.npy"),
        "cov": arr("cov.npy"),
        "panel_mean": arr("panel_mean.npy"),
        "panel_sd": arr("panel_sd.npy"),
        "n_label": arr("n_label.npy"),
        "neigh": arr("neigh.npy"),
        "mu": arr("mu.npy"),
        "mu_n": arr("mu_n.npy"),
        "sin": arr("sin.npy"),
        "snn": arr("snn.npy"),
        "solvable": arr("solvable.npy"),
    }


def choose_refs(sample_ids, groups, mean_cov, n_refs: int, exclude: str | None) -> list[int]:
    idx = [i for i, sid in enumerate(sample_ids) if sid != exclude]
    if n_refs and n_refs < len(idx):
        print(
            "warning: a panel of 50 deepest lineage-balanced libraries failed for myeloid "
            "targets on chromosome 21. The default is the full manifest.",
            flush=True,
        )
        order = sorted(idx, key=lambda i: -mean_cov[i])
        by: dict[str, list[int]] = {}
        for i in order:
            by.setdefault(groups[i] or "ungrouped", []).append(i)
        chosen: list[int] = []
        while len(chosen) < n_refs and any(by.values()):
            for group in sorted(by):
                if by[group] and len(chosen) < n_refs:
                    chosen.append(by[group].pop(0))
        return chosen
    return idx
