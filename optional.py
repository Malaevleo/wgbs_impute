"""Optional imputers. None of these run unless named in --method."""

from __future__ import annotations

import numpy as np


def neighbor_mean(q_beta: np.ndarray, neigh: np.ndarray, panel_mean: float) -> tuple[float, int]:
    nb = neigh[neigh >= 0]
    if len(nb) == 0:
        return panel_mean, 0
    vals = q_beta[nb]
    fin = vals[np.isfinite(vals)]
    if len(fin) == 0 or not np.isfinite(panel_mean):
        return panel_mean, 0
    return float(fin.mean() * 100.0), int(len(fin))


def predict_local(beta, neigh, q_beta, sites, *, iterations, depth, seed, min_ok) -> np.ndarray:
    """Per-site CatBoost. NaNs stay missing; they are not filled with the column mean."""
    try:
        from catboost import CatBoostRegressor, Pool
    except ImportError as exc:
        raise ValueError("CatBoost is required for local/global; install wgbs-impute[ml]") from exc

    out = np.full(len(sites), np.nan)
    for t, i in enumerate(sites):
        nb = neigh[i]
        nb = nb[nb >= 0]
        if len(nb) < 2:
            continue
        y = np.asarray(beta[i], dtype=np.float64)
        x = np.asarray(beta[nb], dtype=np.float64).T
        ok = np.isfinite(y)
        if int(ok.sum()) < min_ok or float(np.nanstd(y)) < 1e-4:
            continue
        x_ok = x[ok]
        keep = np.isfinite(x_ok).any(axis=0)
        if int(keep.sum()) < 2:
            continue
        x_ok = x_ok[:, keep]
        xp = np.asarray(q_beta[nb], dtype=np.float64)[keep][None, :]
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=iterations,
            learning_rate=0.08,
            depth=depth,
            l2_leaf_reg=10.0,
            min_data_in_leaf=3,
            random_seed=seed + int(i),
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(Pool(x_ok, y[ok]))
        out[t] = float(np.clip(model.predict(xp)[0], 0.0, 1.0) * 100.0)
    return out


def predict_global(
    pos,
    meth,
    cov,
    beta10,
    q_meth,
    q_cov,
    sites,
    *,
    max_train,
    refnn_k,
    seed,
    threads,
    iterations: int = 500,
) -> np.ndarray:
    """One CatBoost across sites, trained on panel libraries only."""
    try:
        from catboost import CatBoostRegressor, Pool
    except ImportError as exc:
        raise ValueError("CatBoost is required for local/global; install wgbs-impute[ml]") from exc

    n, n_ref = beta10.shape
    if n_ref < 3 or n < 30:
        return np.full(len(sites), np.nan)
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for j in range(n_ref):
        others = [t for t in range(n_ref) if t != j]
        ref_b = np.asarray(beta10[:, others], dtype=np.float64)
        y = np.asarray(beta10[:, j], dtype=np.float64)
        x, complete = _features(pos, meth[:, j], cov[:, j], ref_b, refnn_k)
        ok = complete & np.isfinite(y)
        pool = np.flatnonzero(ok)
        if len(pool) > max(50, max_train // n_ref):
            pool = rng.choice(pool, size=max(50, max_train // n_ref), replace=False)
        if len(pool):
            xs.append(x[pool])
            ys.append(y[pool])
    if not xs:
        return np.full(len(sites), np.nan)
    x_tr = np.vstack(xs)
    y_tr = np.concatenate(ys)
    if len(y_tr) > max_train:
        take = rng.choice(len(y_tr), size=max_train, replace=False)
        x_tr, y_tr = x_tr[take], y_tr[take]
    if len(y_tr) < 200:
        return np.full(len(sites), np.nan)
    med = np.nanmedian(x_tr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_tr = _fill(x_tr, med)
    ref_b = np.asarray(beta10, dtype=np.float64)
    x_q, _ = _features(pos, q_meth, q_cov, ref_b, refnn_k)
    x_q = _fill(x_q[np.asarray(sites)], med)
    n_va = max(1, int(round(0.1 * len(y_tr))))
    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=iterations,
        learning_rate=0.1,
        depth=6,
        random_seed=seed,
        thread_count=threads,
        early_stopping_rounds=10,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(Pool(x_tr[n_va:], y_tr[n_va:]), eval_set=Pool(x_tr[:n_va], y_tr[:n_va]), use_best_model=True)
    pred = np.clip(model.predict(x_q), 0.0, 1.0) * 100.0
    return pred


def _beta_row(meth, cov) -> np.ndarray:
    out = np.full(len(cov), np.nan)
    ok = cov >= 1
    out[ok] = meth[ok] / np.maximum(cov[ok], 1.0)
    return out


def _fill(x, med):
    out = np.array(x, dtype=np.float64, copy=True)
    for j in range(out.shape[1]):
        bad = ~np.isfinite(out[:, j])
        if bad.any():
            out[bad, j] = med[j]
    return out


def _features(pos, meth, cov, ref_b, k):
    """17 columns plus refNN. Correlation uses the same sites on both sides."""
    y_nb = _beta_row(np.asarray(meth, dtype=np.float64), np.asarray(cov, dtype=np.float64))
    with np.errstate(all="ignore"):
        sample_avg = np.nanmean(ref_b, axis=1)
        pct = np.nanpercentile(ref_b, [10, 25, 50, 75, 90], axis=1)
        ref_sd = np.nanstd(ref_b, axis=1, ddof=1)
    up_b, up_d, dn_b, dn_d = _knn(pos, y_nb, 2)
    local_mean, local_sd = _window(pos, y_nb, 500)
    miss = ~np.isfinite(local_mean) & np.isfinite(sample_avg)
    local_mean = local_mean.copy()
    local_mean[miss] = sample_avg[miss]
    local_sd = np.where(np.isfinite(local_sd), local_sd, 0.0)
    x17 = np.column_stack(
        [
            sample_avg,
            up_b[0],
            dn_b[0],
            up_d[0],
            dn_d[0],
            up_b[1],
            dn_b[1],
            up_d[1],
            dn_d[1],
            local_mean,
            local_sd,
            np.log1p(np.asarray(cov, dtype=np.float64)),
            pct[2],
            ref_sd,
            pct[3] - pct[1],
            pct[0],
            pct[4],
        ]
    )
    nn = _refnn(y_nb, ref_b, k)
    complete = np.isfinite(x17[:, :5]).all(axis=1)
    return np.column_stack([x17, nn]), complete


def _knn(pos, beta, k):
    n = len(pos)
    up_b = [np.full(n, np.nan) for _ in range(k)]
    up_d = [np.full(n, np.nan) for _ in range(k)]
    dn_b = [np.full(n, np.nan) for _ in range(k)]
    dn_d = [np.full(n, np.nan) for _ in range(k)]
    hist: list[int] = []
    for i in range(n):
        for t in range(min(k, len(hist))):
            oi = hist[-(t + 1)]
            up_b[t][i] = beta[oi]
            up_d[t][i] = pos[i] - pos[oi]
        if np.isfinite(beta[i]):
            hist.append(i)
            hist = hist[-k:]
    hist = []
    for i in range(n - 1, -1, -1):
        for t in range(min(k, len(hist))):
            oi = hist[-(t + 1)]
            dn_b[t][i] = beta[oi]
            dn_d[t][i] = pos[oi] - pos[i]
        if np.isfinite(beta[i]):
            hist.append(i)
            hist = hist[-k:]
    return up_b, up_d, dn_b, dn_d


def _window(pos, beta, radius):
    n = len(pos)
    wmean = np.full(n, np.nan)
    wsd = np.full(n, np.nan)
    obs = np.flatnonzero(np.isfinite(beta))
    if len(obs) == 0:
        return wmean, wsd
    obs_pos, obs_beta = pos[obs], beta[obs]
    j_lo = j_hi = 0
    for i in range(n):
        while j_hi < len(obs) and obs_pos[j_hi] <= pos[i] + radius:
            j_hi += 1
        while j_lo < j_hi and obs_pos[j_lo] < pos[i] - radius:
            j_lo += 1
        if j_hi <= j_lo:
            continue
        idxs = np.arange(j_lo, j_hi)
        idxs = idxs[obs[idxs] != i]
        if len(idxs) == 0:
            continue
        d = np.maximum(np.abs(obs_pos[idxs] - pos[i]).astype(np.float64), 1.0)
        w = 1.0 / d
        b = obs_beta[idxs]
        mu = float(np.average(b, weights=w))
        wmean[i] = mu
        wsd[i] = float(np.sqrt(np.average((b - mu) ** 2, weights=w))) if len(b) >= 2 else 0.0
    return wmean, wsd


def _refnn(y_nb, ref_b, k, min_overlap=30):
    n, n_ref = ref_b.shape
    sims = np.full(n_ref, np.nan)
    for j in range(n_ref):
        both = np.isfinite(y_nb) & np.isfinite(ref_b[:, j])
        if both.sum() < min_overlap:
            continue
        a = y_nb[both]
        b = ref_b[both, j]
        a = a - a.mean()
        b = b - b.mean()
        da = np.sqrt(np.mean(a * a))
        db = np.sqrt(np.mean(b * b))
        if da < 1e-8 or db < 1e-8:
            continue
        sims[j] = float(np.mean(a * b) / (da * db))
    order = np.argsort(-np.nan_to_num(sims, nan=-2.0))
    valid = [int(j) for j in order if np.isfinite(sims[j])][:k]
    feats = np.full((n, k + 2), np.nan)
    for t, j in enumerate(valid):
        feats[:, t] = ref_b[:, j]
    if valid:
        feats[:, k] = np.nanmean(feats[:, : min(3, len(valid))], axis=1)
        feats[:, k + 1] = np.nanmean(feats[:, : min(5, len(valid))], axis=1)
    return feats


def segment_blocks(pos, max_gap=2000, max_sites=50, max_span=25000):
    n = len(pos)
    bid = np.empty(n, dtype=np.int32)
    b = start = 0
    bid[0] = 0
    for i in range(1, n):
        if pos[i] - pos[i - 1] > max_gap or (i - start) >= max_sites or pos[i] - pos[start] > max_span:
            b += 1
            start = i
        bid[i] = b
    return bid


def _kmeans(x, k, rng):
    n, p = x.shape
    k = int(min(k, n))
    if k <= 1 or n < 4:
        return x.mean(axis=0, keepdims=True)
    centers = np.empty((k, p))
    centers[0] = x[int(rng.integers(n))]
    closest = np.full(n, np.inf)
    for j in range(1, k):
        closest = np.minimum(closest, ((x - centers[j - 1]) ** 2).sum(axis=1))
        tot = float(closest.sum())
        if tot <= 0:
            centers[j:] = x[int(rng.integers(n))]
            break
        centers[j] = x[int(rng.choice(n, p=closest / tot))]
    lab = np.zeros(n, dtype=np.int32)
    for _ in range(15):
        lab = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        for j in range(k):
            m = lab == j
            if m.any():
                centers[j] = x[m].mean(axis=0)
    keep = np.array([(lab == j).any() for j in range(k)])
    return np.clip(centers[keep], 1e-3, 1 - 1e-3)


def predict_comet(pos, beta, q_meth, q_cov, panel_mean, hide, seed=42) -> np.ndarray:
    """Block archetypes. Weights come from sites that are not being imputed."""
    beta = np.asarray(beta, dtype=np.float64)
    block = segment_blocks(pos)
    rng = np.random.default_rng(seed)
    out = np.asarray(panel_mean, dtype=np.float64).copy()
    for b in range(int(block.max()) + 1):
        rows = np.flatnonzero(block == b)
        block_beta = beta[rows]
        mu = np.nanmean(block_beta, axis=1)
        mu = np.where(np.isfinite(mu), mu, 0.5)
        filled = np.where(np.isfinite(block_beta), block_beta, mu[:, None])
        good = np.isfinite(block_beta).any(axis=0)
        if good.sum() < 4 or len(rows) < 3:
            continue
        centers = _kmeans(filled[:, good].T, 6, rng)
        use = (~hide[rows]) & (q_cov[rows] >= 1)
        if int(use.sum()) < 2:
            continue
        x = np.clip(q_meth[rows][use] / np.maximum(q_cov[rows][use], 1.0), 1e-4, 1 - 1e-4)
        w = _simplex(centers[:, use], x, q_cov[rows][use])
        pred = centers.T @ w
        out[rows] = np.clip(pred, 0.0, 1.0) * 100.0
    return out


def _simplex(p, x, wts, n_iter=40):
    """p is (k, n_obs)."""
    k = p.shape[0]
    w = np.ones(k) / k
    sw = np.sqrt(np.maximum(wts, 1e-6))
    pw = p * sw
    xw = x * sw
    gram = pw @ pw.T
    step = 1.0 / max(float(np.trace(gram)), 1e-3)
    for _ in range(n_iter):
        w = np.maximum(w - step * (gram @ w - pw @ xw), 0.0)
        s = float(w.sum())
        w = w / s if s > 1e-12 else np.ones(k) / k
    return w


def predict_hmm(pos, beta, q_meth, q_cov, hide, lam=1.0 / 20000.0) -> np.ndarray:
    """Li–Stephens copying HMM. Hidden sites contribute no emission."""
    beta = np.where(np.isfinite(beta), beta, 0.5)
    t_len, n_s = beta.shape
    br = np.clip(beta, 1e-4, 1 - 1e-4)
    e = np.ones((t_len, n_s))
    use = (q_cov >= 1) & (~hide)
    if use.any():
        m = q_meth[use]
        c = q_cov[use]
        p = br[use]
        ll = m[:, None] * np.log(p) + (c - m)[:, None] * np.log(1.0 - p)
        ll -= ll.max(axis=1, keepdims=True)
        e[use] = np.maximum(np.exp(ll), 1e-300)
    alpha = np.empty((t_len, n_s))
    alpha[0] = e[0] / n_s
    alpha[0] /= alpha[0].sum() + 1e-300
    stay = np.ones(t_len)
    for t in range(1, t_len):
        a = min(max(float(np.exp(-lam * max(int(pos[t] - pos[t - 1]), 0))), 0.0), 0.999999)
        stay[t] = a
        sw = (1.0 - a) / max(n_s - 1, 1)
        pred = (a - sw) * alpha[t - 1] + sw * float(alpha[t - 1].sum())
        alpha[t] = pred * e[t]
        alpha[t] /= alpha[t].sum() + 1e-300
    beta_b = np.ones((t_len, n_s))
    for t in range(t_len - 2, -1, -1):
        a = stay[t + 1]
        sw = (1.0 - a) / max(n_s - 1, 1)
        nxt = beta_b[t + 1] * e[t + 1]
        bwd = (a - sw) * nxt + sw * float(nxt.sum())
        beta_b[t] = bwd / (bwd.sum() + 1e-300)
    gamma = alpha * beta_b
    gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
    return np.clip((gamma * br).sum(axis=1), 0.0, 1.0) * 100.0


def predict_kalman(pos, panel_mean, panel_sd, q_meth, q_cov, hide, length_bp=800.0) -> np.ndarray:
    """RTS smoother with the panel mean as the prior. Hidden sites are not observations."""
    t_len = len(pos)
    mu = np.clip(np.where(np.isfinite(panel_mean), panel_mean / 100.0, 0.5), 1e-3, 1 - 1e-3)
    sig2 = np.maximum((np.asarray(panel_sd, dtype=np.float64) / 100.0) ** 2, 1e-4)
    x_f = np.empty(t_len)
    p_f = np.empty(t_len)
    x_p = np.empty(t_len)
    p_p = np.empty(t_len)
    a_t = np.zeros(t_len)
    x_p[0], p_p[0] = mu[0], sig2[0]
    for t in range(t_len):
        if t > 0:
            a = min(max(float(np.exp(-max(int(pos[t] - pos[t - 1]), 0) / length_bp)), 0.0), 0.999)
            a_t[t] = a
            x_p[t] = mu[t] + a * (x_f[t - 1] - mu[t - 1])
            p_p[t] = a * a * p_f[t - 1] + (1.0 - a * a) * sig2[t]
        if q_cov[t] >= 1 and not hide[t]:
            y = float(np.clip(q_meth[t] / max(float(q_cov[t]), 1.0), 1e-4, 1 - 1e-4))
            r = max(y * (1.0 - y), 0.05) / max(float(q_cov[t]), 1.0)
            gain = p_p[t] / (p_p[t] + r)
            x_f[t] = x_p[t] + gain * (y - x_p[t])
            p_f[t] = (1.0 - gain) * p_p[t]
        else:
            x_f[t], p_f[t] = x_p[t], p_p[t]
    x_s = x_f.copy()
    for t in range(t_len - 2, -1, -1):
        if p_f[t] < 1e-12:
            continue
        g = a_t[t + 1] * p_f[t] / max(p_p[t + 1], 1e-12)
        x_s[t] = x_f[t] + g * (x_s[t + 1] - x_p[t + 1])
    return np.clip(x_s, 0.0, 1.0) * 100.0
