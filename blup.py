"""Context BLUP with pairwise-complete covariance, and the fusion weight."""

from __future__ import annotations

import numpy as np

MIN_OVERLAP = 15


def blup_gates(n_take: int) -> tuple[int, int]:
    """Same gates as run_panel_size_ablation.blup_gates."""
    if n_take < 10:
        min_ok = max(3, n_take // 2 + 1)
        min_ov = max(2, min(n_take - 1, n_take // 2 + 1))
        return min_ok, min_ov
    return max(5, min(12, n_take // 2 + 1)), max(5, min(MIN_OVERLAP, n_take - 1))


def calibrate(pred, truth) -> float:
    """Same weight as run_panel_size_ablation.calibrate. Returns κ in [1, 80]."""
    ok = np.isfinite(pred) & np.isfinite(truth)
    if ok.sum() < 50:
        return 8.0
    pi = np.clip(np.asarray(pred, dtype=np.float64)[ok] / 100.0, 1e-3, 1 - 1e-3)
    t = np.asarray(truth, dtype=np.float64)[ok] / 100.0
    sigma2 = float(max(np.mean((pi - t) ** 2), 1e-4))
    return float(np.clip(np.median(pi * (1.0 - pi) / sigma2), 1.0, 80.0))


def fuse(pred_pct, meth, cov, kappa: float) -> np.ndarray:
    """Beta-binomial posterior mean. Coverage 0 returns the context prediction."""
    pred = np.asarray(pred_pct, dtype=np.float64)
    meth_a = np.asarray(meth, dtype=np.float64)
    cov_a = np.asarray(cov, dtype=np.float64)
    pi = np.clip(pred / 100.0, 0.0, 1.0)
    m = np.clip(meth_a, 0.0, cov_a)
    denom = float(kappa) + cov_a
    out = np.where(denom > 0, 100.0 * (float(kappa) * pi + m) / np.maximum(denom, 1e-12), pred)
    return out


def pairwise_shrunk_cov(columns: np.ndarray, shrink: float) -> tuple[np.ndarray, np.ndarray]:
    """Column means and shrunk covariance. A pair contributes only when both are observed.

    `columns` is (n_ref, p). Off-diagonals are multiplied by (1 - shrink). The
    diagonal is left as the pairwise variance, then 1e-6 is added. Missing
    entries are not filled.
    """
    n, p = columns.shape
    mu = np.nanmean(columns, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.5)
    cov = np.zeros((p, p), dtype=np.float64)
    finite = np.isfinite(columns)
    for a in range(p):
        for b in range(a, p):
            both = finite[:, a] & finite[:, b]
            n_both = int(both.sum())
            if n_both < 2:
                c = 0.0
            else:
                da = columns[both, a] - columns[both, a].mean()
                db = columns[both, b] - columns[both, b].mean()
                c = float(np.dot(da, db) / (n_both - 1))
            cov[a, b] = c
            cov[b, a] = c
    diag = np.diag(cov).copy()
    cov *= 1.0 - shrink
    cov[np.diag_indices(p)] = diag
    cov[np.diag_indices(p)] += 1e-6
    return mu, cov


def select_neighbors(
    pos: np.ndarray,
    beta: np.ndarray,
    window: int,
    k: int,
    min_overlap: int,
) -> np.ndarray:
    """Top-k neighbors inside ±window by |pairwise Pearson|. Shape (n, k), -1 padded.

    The overlap filter applies even when the window holds fewer than k CpGs.
    Neighbors are indices into `pos` / `beta` and therefore stay on this chromosome.
    """
    pos = np.asarray(pos)
    n = len(pos)
    out = np.full((n, k), -1, dtype=np.int32)
    if n == 0 or k <= 0:
        return out
    for i in range(n):
        lo = int(np.searchsorted(pos, pos[i] - window, side="left"))
        hi = int(np.searchsorted(pos, pos[i] + window, side="right"))
        cand = np.arange(lo, hi, dtype=np.int32)
        cand = cand[cand != i]
        if len(cand) == 0:
            continue
        corr = _abs_corr(beta[i], beta[cand], min_overlap)
        keep = np.flatnonzero(corr > 0)
        if len(keep) == 0:
            continue
        order = keep[np.argsort(-corr[keep])][:k]
        out[i, : len(order)] = cand[order]
    return out


def _abs_corr(y: np.ndarray, x: np.ndarray, min_overlap: int) -> np.ndarray:
    """|Pearson| of one site against each candidate row. Candidates with too little overlap are 0."""
    both = np.isfinite(x) & np.isfinite(y)[None, :]
    n_ov = both.sum(axis=1)
    mu_y = np.divide((np.where(both, y, 0.0)).sum(axis=1), np.maximum(n_ov, 1))
    mu_x = np.divide((np.where(both, x, 0.0)).sum(axis=1), np.maximum(n_ov, 1))
    dy = np.where(both, y[None, :] - mu_y[:, None], 0.0)
    dx = np.where(both, x - mu_x[:, None], 0.0)
    n_m1 = np.maximum(n_ov - 1, 1)
    cov = (dy * dx).sum(axis=1) / n_m1
    vy = (dy * dy).sum(axis=1) / n_m1
    vx = (dx * dx).sum(axis=1) / n_m1
    corr = np.zeros(len(x), dtype=np.float64)
    ok = (n_ov >= min_overlap) & (vy > 1e-12) & (vx > 1e-12)
    corr[ok] = np.abs(cov[ok] / np.sqrt(vy[ok] * vx[ok]))
    corr[~np.isfinite(corr)] = 0.0
    return corr


def site_factors(
    beta: np.ndarray,
    neighbors: np.ndarray,
    shrink: float,
    min_ok: int,
) -> dict[str, np.ndarray]:
    """Query-independent BLUP pieces. `beta` is (n_sites, n_refs), fraction or NaN."""
    n, k = neighbors.shape
    mu = np.full(n, np.nan, dtype=np.float32)
    mu_n = np.zeros((n, k), dtype=np.float32)
    sin = np.zeros((n, k), dtype=np.float32)
    snn = np.zeros((n, k, k), dtype=np.float32)
    solvable = np.zeros(n, dtype=bool)
    with np.errstate(all="ignore"):
        panel_mean = np.nanmean(beta, axis=1) * 100.0
        panel_sd = np.nanstd(beta, axis=1, ddof=1) * 100.0
        n_label = np.sum(np.isfinite(beta), axis=1).astype(np.int16)
    panel_mean = np.where(np.isfinite(panel_mean), panel_mean, np.nan).astype(np.float32)
    panel_sd = np.where(np.isfinite(panel_sd), panel_sd, 0.0).astype(np.float32)
    for i in range(n):
        nb = neighbors[i]
        nb = nb[nb >= 0]
        if len(nb) < 2 or int(n_label[i]) < min_ok:
            continue
        cols = np.column_stack([np.asarray(beta[i], dtype=np.float64), np.asarray(beta[nb], dtype=np.float64).T])
        # Drop references with no label. Neighbor holes stay pairwise.
        ok_y = np.isfinite(cols[:, 0])
        mu_v, cov = pairwise_shrunk_cov(cols[ok_y], shrink)
        m = len(nb)
        mu[i] = np.float32(mu_v[0])
        mu_n[i, :m] = mu_v[1:]
        sin[i, :m] = cov[0, 1:]
        snn[i, :m, :m] = cov[1:, 1:]
        solvable[i] = True
    return {
        "panel_mean": panel_mean,
        "panel_sd": panel_sd,
        "n_label": n_label,
        "mu": mu,
        "mu_n": mu_n,
        "sin": sin,
        "snn": snn,
        "solvable": solvable,
    }


def context_one(
    mu_i: float,
    mu_n: np.ndarray,
    sin: np.ndarray,
    snn: np.ndarray,
    q_beta: np.ndarray,
    q_cov: np.ndarray,
    n_label: int,
    min_ok: int,
    noise_floor: float,
    panel_mean: float,
    solvable: bool,
) -> tuple[float, int, bool]:
    """Return (percent, n_query_neighbors_used, solved)."""
    if not solvable or n_label < min_ok or not np.isfinite(panel_mean):
        return float(panel_mean) if np.isfinite(panel_mean) else float("nan"), 0, False
    obs = np.isfinite(q_beta) & (q_cov >= 1)
    n_obs = int(obs.sum())
    if n_obs < 2 or not np.isfinite(mu_i):
        return float(panel_mean), n_obs, False
    x = q_beta[obs].astype(np.float64)
    xc = np.clip(x, 1e-4, 1.0 - 1e-4)
    diag = np.maximum(xc * (1.0 - xc), noise_floor) / np.maximum(q_cov[obs], 1.0)
    a = np.array(snn[np.ix_(obs, obs)], dtype=np.float64, copy=True)
    a.flat[:: a.shape[0] + 1] += diag
    try:
        alpha = np.linalg.solve(a, x - mu_n[obs])
    except np.linalg.LinAlgError:
        return float(panel_mean), n_obs, False
    pi = float(mu_i) + float(sin[obs] @ alpha)
    return 100.0 * float(np.clip(pi, 0.0, 1.0)), n_obs, True
