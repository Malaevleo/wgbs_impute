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


def meanfill_shrunk_cov(columns: np.ndarray, shrink: float) -> tuple[np.ndarray, np.ndarray]:
    """Research covariance: missing entries are replaced by the column mean, then shrunk.

    This is the estimator behind the published chromosome-21 scores. Columns with
    no observations get mean 0.5. Diagonal variances are floored at 1e-4 before
    the shrink, and 1e-6 is added after it.
    """
    col_ok = np.isfinite(columns).any(axis=0)
    mu = np.where(col_ok, np.nanmean(columns, axis=0), 0.5)
    mu = np.clip(np.where(np.isfinite(mu), mu, 0.5), 0.0, 1.0)
    filled = np.clip(np.where(np.isfinite(columns), columns, mu), 0.0, 1.0)
    n = filled.shape[0]
    centered = filled - mu
    scatter = (centered.T @ centered) / max(n - 1, 1)
    diag = np.maximum(np.diag(scatter).copy(), 1e-4)
    cov = (1.0 - shrink) * scatter + shrink * np.diag(diag)
    cov.flat[:: cov.shape[0] + 1] += 1e-6
    return mu, cov


def znorm_rows(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre each row on its observed entries. Missing entries become 0."""
    obs = np.isfinite(mat)
    counts = obs.sum(axis=1, keepdims=True)
    filled = np.where(obs, mat, 0.0)
    means = filled.sum(axis=1, keepdims=True) / np.maximum(counts, 1)
    centred = np.where(obs, mat - means, 0.0)
    norms = np.sqrt((centred ** 2).sum(axis=1, keepdims=True))
    z = np.zeros_like(centred)
    good = norms[:, 0] > 1e-8
    z[good] = centred[good] / norms[good]
    return z, obs.astype(np.float64)


def research_neighbors(
    pos: np.ndarray,
    beta: np.ndarray,
    window: int,
    k: int,
    min_overlap: int,
    sites: np.ndarray | None = None,
) -> np.ndarray:
    """Neighbors ranked by the cosine of z-scored rows, as in the research code.

    A window with k or fewer CpGs keeps all of them and does not apply the overlap
    filter. `sites` limits which rows are filled; other rows stay -1.
    """
    pos = np.asarray(pos)
    n = len(pos)
    out = np.full((n, k), -1, dtype=np.int32)
    if n == 0 or k <= 0:
        return out
    z, mask = znorm_rows(np.asarray(beta, dtype=np.float64))
    lo_idx = np.searchsorted(pos, pos - window, side="left")
    hi_idx = np.searchsorted(pos, pos + window, side="right")
    which = np.arange(n) if sites is None else np.asarray(sites, dtype=np.int64)
    for i in which:
        lo, hi = int(lo_idx[i]), int(hi_idx[i])
        cand = np.arange(lo, hi, dtype=np.int64)
        cand = cand[cand != i]
        if len(cand) == 0:
            continue
        if len(cand) <= k:
            out[i, : len(cand)] = cand
            continue
        with np.errstate(all="ignore"):
            overlap = mask[i] @ mask[cand].T
            corr = z[i] @ z[cand].T
        corr = np.where((overlap >= min_overlap) & np.isfinite(corr), corr, 0.0)
        take = np.argpartition(np.abs(corr), -k)[-k:]
        take = take[np.argsort(-np.abs(corr[take]))]
        out[i, :k] = cand[take]
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
    beta_x: np.ndarray | None = None,
    filled: bool = False,
) -> dict[str, np.ndarray]:
    """Query-independent BLUP pieces. `beta` is (n_sites, n_refs), fraction or NaN.

    `filled=True` is the research covariance: neighbor columns come from `beta_x`
    (coverage at least 1) and missing entries are mean-filled.
    """
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
        xsrc = beta if beta_x is None else beta_x
        cols = np.column_stack([
            np.asarray(beta[i], dtype=np.float64),
            np.asarray(xsrc[nb], dtype=np.float64).T,
        ])
        ok_y = np.isfinite(cols[:, 0])
        cov_fn = meanfill_shrunk_cov if filled else pairwise_shrunk_cov
        mu_v, cov = cov_fn(cols[ok_y], shrink)
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


def predict_research(
    y_refs, x_nb, query_beta, query_cov, eval_idx, neighbors, panel_mean,
    shrink=0.25, min_ok=12, floor=0.05,
) -> np.ndarray:
    """Research BLUP. Same algebra as audit_opus.blup_predict_v."""
    out = np.full(len(eval_idx), np.nan)
    for s, i in enumerate(eval_idx):
        neigh = neighbors[s]
        fb = panel_mean[i]
        if len(neigh) < 2:
            out[s] = fb
            continue
        y = y_refs[i]
        x = x_nb[neigh].T
        ok_y = np.isfinite(y)
        if int(ok_y.sum()) < min_ok:
            out[s] = fb
            continue
        mu_yx, cov_yx = meanfill_shrunk_cov(np.column_stack([y[ok_y], x[ok_y]]), shrink)
        mu_i, mu_n = float(mu_yx[0]), mu_yx[1:]
        sigma_in = cov_yx[0, 1:]
        sigma_nn = cov_yx[1:, 1:]
        x_q = query_beta[neigh]
        c_q = query_cov[neigh].astype(np.float64)
        obs = np.isfinite(x_q) & (c_q >= 1)
        if int(obs.sum()) < 2:
            out[s] = 100.0 * float(np.clip(mu_i, 0.0, 1.0))
            continue
        xv = np.clip(x_q[obs], 1e-4, 1.0 - 1e-4)
        diag = np.maximum(xv * (1.0 - xv), floor) / np.maximum(c_q[obs], 1.0)
        a = sigma_nn[np.ix_(obs, obs)].copy()
        a.flat[:: a.shape[0] + 1] += diag
        try:
            alpha = np.linalg.solve(a, xv - mu_n[obs])
        except np.linalg.LinAlgError:
            out[s] = 100.0 * float(np.clip(mu_i, 0.0, 1.0))
            continue
        out[s] = 100.0 * float(np.clip(mu_i + float(sigma_in[obs] @ alpha), 0.0, 1.0))
    return out


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
    clip_query: bool = False,
) -> tuple[float, int, bool]:
    """Return (percent, n_query_neighbors_used, solved).

    `clip_query=True` is the research solve: the query value is clipped before
    the residual, and fewer than two observed neighbors returns the clipped
    reference mean of the site.
    """
    if not solvable or n_label < min_ok or not np.isfinite(panel_mean):
        return float(panel_mean) if np.isfinite(panel_mean) else float("nan"), 0, False
    obs = np.isfinite(q_beta) & (q_cov >= 1)
    n_obs = int(obs.sum())
    if n_obs < 2 or not np.isfinite(mu_i):
        if clip_query and np.isfinite(mu_i):
            return 100.0 * float(np.clip(mu_i, 0.0, 1.0)), n_obs, True
        return float(panel_mean), n_obs, False
    x = q_beta[obs].astype(np.float64)
    if clip_query:
        x = np.clip(x, 1e-4, 1.0 - 1e-4)
    xc = x if clip_query else np.clip(x, 1e-4, 1.0 - 1e-4)
    diag = np.maximum(xc * (1.0 - xc), noise_floor) / np.maximum(q_cov[obs], 1.0)
    a = np.array(snn[np.ix_(obs, obs)], dtype=np.float64, copy=True)
    a.flat[:: a.shape[0] + 1] += diag
    try:
        alpha = np.linalg.solve(a, x - mu_n[obs])
    except np.linalg.LinAlgError:
        return float(panel_mean), n_obs, False
    pi = float(mu_i) + float(sin[obs] @ alpha)
    return 100.0 * float(np.clip(pi, 0.0, 1.0)), n_obs, True
